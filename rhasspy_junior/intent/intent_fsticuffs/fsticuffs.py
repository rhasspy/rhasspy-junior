"""Recognition functions for sentences using JSGF graphs."""
import base64
import itertools
import random
import time
import typing
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
from lingua_franca.parse import extract_datetime, extract_duration, extract_number

from .intent import Entity, Intent, Recognition, RecognitionResult
from .jsgf import Word
from .jsgf_graph import get_start_end_nodes
from .utils import pairwise

# -----------------------------------------------------------------------------

PathNodeType = typing.Union[int, typing.Tuple[int, typing.List[str]]]
PathType = typing.List[PathNodeType]


def recognize(
    tokens: typing.Union[str, typing.List[str]],
    graph: nx.DiGraph,
    fuzzy: bool = True,
    stop_words: typing.Optional[typing.Set[str]] = None,
    intent_filter: typing.Optional[typing.Callable[[str], bool]] = None,
    word_transform: typing.Optional[typing.Callable[[str], str]] = None,
    converters: typing.Optional[
        typing.Dict[str, typing.Callable[..., typing.Any]]
    ] = None,
    extra_converters: typing.Optional[
        typing.Dict[str, typing.Callable[..., typing.Any]]
    ] = None,
    **search_args,
) -> typing.List[Recognition]:
    """Recognize one or more intents from tokens or a sentence."""
    start_time = time.perf_counter()

    if isinstance(tokens, str):
        # Assume whitespace separation
        tokens = tokens.split()

    if fuzzy:
        # Fuzzy recognition
        best_fuzzy = best_fuzzy_cost(
            paths_fuzzy(
                tokens,
                graph,
                stop_words=stop_words,
                intent_filter=intent_filter,
                word_transform=word_transform,
                **search_args,
            )
        )

        end_time = time.perf_counter()
        if best_fuzzy:
            recognitions = []

            # Gather all successful fuzzy paths
            for fuzzy_result in best_fuzzy:
                result, recognition = path_to_recognition(
                    fuzzy_result.node_path,
                    graph,
                    cost=fuzzy_result.cost,
                    converters=converters,
                    extra_converters=extra_converters,
                )
                if result == RecognitionResult.SUCCESS:
                    assert recognition is not None
                    recognition.recognize_seconds = end_time - start_time
                    recognitions.append(recognition)

            return recognitions
    else:
        # Strict recognition
        paths = list(
            paths_strict(
                tokens,
                graph,
                intent_filter=intent_filter,
                word_transform=word_transform,
                **search_args,
            )
        )
        if (not paths) and stop_words:
            # Try again by excluding stop words
            tokens = [t for t in tokens if t not in stop_words]
            paths = list(
                paths_strict(
                    tokens,
                    graph,
                    exclude_tokens=stop_words,
                    intent_filter=intent_filter,
                    word_transform=word_transform,
                    **search_args,
                )
            )

        end_time = time.perf_counter()
        recognitions = []
        for path in paths:
            result, recognition = path_to_recognition(
                path, graph, converters=converters, extra_converters=extra_converters
            )
            if result == RecognitionResult.SUCCESS:
                assert recognition is not None
                recognition.recognize_seconds = end_time - start_time
                recognitions.append(recognition)

        return recognitions

    # No results
    return []


# -----------------------------------------------------------------------------


def paths_strict(
    tokens: typing.List[str],
    graph: nx.DiGraph,
    exclude_tokens: typing.Optional[typing.Set[str]] = None,
    max_paths: typing.Optional[int] = None,
    intent_filter: typing.Optional[typing.Callable[[str], bool]] = None,
    word_transform: typing.Optional[typing.Callable[[str], str]] = None,
) -> typing.Iterable[PathType]:
    """Match a single path from the graph exactly if possible."""
    if not tokens:
        return []

    intent_filter = intent_filter or (lambda x: True)
    word_transform = word_transform or (lambda x: x)

    # node -> attrs
    n_data = graph.nodes(data=True)

    # start state
    start_node, _ = get_start_end_nodes(graph)
    assert start_node is not None

    # Number of matching paths found
    paths_found: int = 0

    # Do breadth-first search.
    # Queue contains items of the form:
    # * current node (int)
    # * current path (int, str?) - node, matching input token
    # * remaining input tokens
    node_queue: typing.List[typing.Tuple[int, PathType, typing.List[str]]] = [
        (start_node, [], tokens)
    ]

    while node_queue:
        current_node, current_path, current_tokens = node_queue.pop(0)
        is_final = n_data[current_node].get("final", False)
        if is_final and (not current_tokens):
            # Reached final state
            paths_found += 1
            yield current_path

            if max_paths and (paths_found >= max_paths):
                break

        for next_node, edge_data in graph[current_node].items():
            next_path = list(current_path)
            next_tokens = list(current_tokens)

            ilabel = edge_data.get("ilabel", "")
            olabel = edge_data.get("olabel", "")
            matching_tokens: typing.List[str] = []

            if olabel[:9] == "__label__":
                intent_name = olabel[9:]
                if not intent_filter(intent_name):
                    # Skip intent
                    continue

            if ilabel:
                ilabel = word_transform(ilabel)

                if next_tokens:
                    # Failed to match input label
                    if ilabel != word_transform(next_tokens[0]):
                        if (not exclude_tokens) or (ilabel not in exclude_tokens):
                            # Can't exclude
                            continue
                    else:
                        # Token match
                        matching_token = next_tokens.pop(0)
                        matching_tokens.append(matching_token)
                else:
                    # Ran out of tokens
                    continue

            next_path.append((current_node, matching_tokens))

            # Continue search
            node_queue.append((next_node, next_path, next_tokens))

    # No results
    return []


# -----------------------------------------------------------------------------


@dataclass
class FuzzyResult:
    """Single path for fuzzy recognition."""

    intent_name: str
    node_path: PathType
    cost: float


@dataclass
class FuzzyCostInput:
    """Input to fuzzy cost function."""

    ilabel: str
    tokens: typing.List[str]
    stop_words: typing.Set[str]
    word_transform: typing.Optional[typing.Callable[[str], str]] = None
    in_wildcard: bool = False


@dataclass
class FuzzyCostOutput:
    """Output from fuzzy cost function."""

    cost: float
    continue_search: bool = True
    matching_tokens: typing.List[str] = field(default_factory=list)
    in_wildcard: bool = False


def default_fuzzy_cost(cost_input: FuzzyCostInput) -> FuzzyCostOutput:
    """Increases cost when input tokens fail to match graph. Marginal cost for stop words."""
    ilabel = cost_input.ilabel
    cost = 0.0
    tokens = cost_input.tokens
    stop_words = cost_input.stop_words
    word_transform = cost_input.word_transform or (lambda x: x)
    matching_tokens: typing.List[str] = []

    if ilabel:
        if tokens and (ilabel == Word.WILDCARD):
            matching_tokens.append(tokens.pop(0))
            return FuzzyCostOutput(cost=0.1, matching_tokens=matching_tokens)

        ilabel = word_transform(ilabel)
        while tokens and (ilabel != word_transform(tokens[0])):
            bad_token = word_transform(tokens.pop(0))

            if bad_token in stop_words:
                # Marginal cost to ensure paths matching stop words are preferred
                cost += 0.1
            else:
                # Mismatched token
                cost += 1

        if tokens and (ilabel == word_transform(tokens[0])):
            # Consume matching token
            matching_token = tokens.pop(0)
            matching_tokens.append(matching_token)
        else:
            # No matching token
            return FuzzyCostOutput(cost=cost, continue_search=False)

    return FuzzyCostOutput(cost=cost, matching_tokens=matching_tokens)


def paths_fuzzy(
    tokens: typing.List[str],
    graph: nx.DiGraph,
    stop_words: typing.Optional[typing.Set[str]] = None,
    cost_function: typing.Optional[
        typing.Callable[[FuzzyCostInput], FuzzyCostOutput]
    ] = None,
    intent_filter: typing.Optional[typing.Callable[[str], bool]] = None,
    word_transform: typing.Optional[typing.Callable[[str], str]] = None,
) -> typing.Dict[str, typing.List[FuzzyResult]]:
    """Do less strict matching using a cost function and optional stop words."""
    if not tokens:
        return {}

    intent_filter = intent_filter or (lambda x: True)
    cost_function = cost_function or default_fuzzy_cost
    stop_words = stop_words or set()

    # node -> attrs
    n_data = graph.nodes(data=True)

    # start state
    start_node: int = next(n for n, data in n_data if data.get("start"))

    # intent -> [(symbols, cost), (symbols, cost)...]
    intent_symbols_and_costs: typing.Dict[str, typing.List[FuzzyResult]] = defaultdict(
        list
    )

    # Lowest cost so far
    best_cost: float = float(len(n_data))

    # (node, in_tokens, out_path, out_count, cost, intent_name)
    node_queue: typing.List[
        typing.Tuple[int, typing.List[str], PathType, int, float, typing.Optional[str]]
    ] = [(start_node, tokens, [], 0, 0.0, None)]

    # BFS it up
    while node_queue:
        (
            q_node,
            q_in_tokens,
            q_out_path,
            q_out_count,
            q_cost,
            q_intent,
        ) = node_queue.pop(0)
        is_final: bool = n_data[q_node].get("final", False)

        # Update best intent cost on final state.
        # Don't bother reporting intents that failed to consume any tokens.
        if is_final and (q_cost < q_out_count):
            q_intent = q_intent or ""
            best_intent_cost: typing.Optional[float] = None
            best_intent_costs = intent_symbols_and_costs.get(q_intent)
            if best_intent_costs:
                best_intent_cost = best_intent_costs[0].cost

            final_cost = q_cost + len(q_in_tokens)  # remaning tokens count against
            final_path = q_out_path

            if (best_intent_cost is None) or (final_cost < best_intent_cost):
                # Overwrite best cost
                intent_symbols_and_costs[q_intent] = [
                    FuzzyResult(
                        intent_name=q_intent, node_path=final_path, cost=final_cost
                    )
                ]
            elif final_cost == best_intent_cost:
                # Add to existing list
                intent_symbols_and_costs[q_intent].append(
                    (
                        FuzzyResult(
                            intent_name=q_intent, node_path=final_path, cost=final_cost
                        )
                    )
                )

            if final_cost < best_cost:
                # Update best cost so far
                best_cost = final_cost

        if q_cost > best_cost:
            # Can't get any better
            continue

        # Process child edges
        for next_node, edge_data in graph[q_node].items():
            in_label = edge_data.get("ilabel") or ""
            out_label = edge_data.get("olabel") or ""
            next_in_tokens = list(q_in_tokens)
            next_out_path = list(q_out_path)
            next_out_count = q_out_count
            next_cost = q_cost
            next_intent = q_intent

            if out_label:
                if out_label[:9] == "__label__":
                    next_intent = out_label[9:]
                    if not intent_filter(next_intent):
                        # Skip intent
                        continue
                elif out_label[:2] != "__":
                    next_out_count += 1

            cost_output = cost_function(
                FuzzyCostInput(
                    ilabel=in_label,
                    tokens=next_in_tokens,
                    stop_words=stop_words,
                    word_transform=word_transform,
                )
            )

            next_cost += cost_output.cost

            if not cost_output.continue_search:
                continue

            # Extend current path
            next_out_path.append((q_node, cost_output.matching_tokens))

            node_queue.append(
                (
                    next_node,
                    next_in_tokens,
                    next_out_path,
                    next_out_count,
                    next_cost,
                    next_intent,
                )
            )

    return intent_symbols_and_costs


def best_fuzzy_cost(
    intent_symbols_and_costs: typing.Dict[str, typing.List[FuzzyResult]]
) -> typing.List[FuzzyResult]:
    """Return fuzzy results with cost."""
    if not intent_symbols_and_costs:
        return []

    best_cost: typing.Optional[float] = None
    best_results: typing.List[FuzzyResult] = []

    # Find all results with the lowest cost
    for fuzzy_results in intent_symbols_and_costs.values():
        if not fuzzy_results:
            continue

        # All results for a given intent should have the same cost
        if best_cost is None:
            # First result
            best_cost = fuzzy_results[0].cost
            best_results = list(fuzzy_results)
        elif fuzzy_results[0].cost < best_cost:
            # Overwrite
            best_results = list(fuzzy_results)
        elif fuzzy_results[0].cost == best_cost:
            # Add to list
            best_results.extend(fuzzy_results)

    return best_results


# -----------------------------------------------------------------------------


@dataclass
class ConverterInfo:
    """Local info for converter stack in path_to_recognition"""

    # Name + args
    key: str

    # Name of converter
    name: str

    # Optional arguments passed using name,arg1,arg2,...
    args: typing.Optional[typing.List[str]] = None

    # List of raw/substituted tokens
    tokens: typing.List[typing.Tuple[str, str, typing.List[str]]] = field(
        default_factory=list
    )


def path_to_recognition(
    node_path: typing.Iterable[PathNodeType],
    graph: nx.DiGraph,
    cost: typing.Optional[float] = None,
    converters: typing.Optional[
        typing.Dict[str, typing.Callable[..., typing.Any]]
    ] = None,
    extra_converters: typing.Optional[
        typing.Dict[str, typing.Callable[..., typing.Any]]
    ] = None,
) -> typing.Tuple[RecognitionResult, typing.Optional[Recognition]]:
    """Transform node path in graph to an intent recognition object."""
    if not node_path:
        # Empty path indicates failure
        return RecognitionResult.FAILURE, None

    converters = converters or get_default_converters()
    if extra_converters:
        # Merge in extra converters
        converters.update(extra_converters)

    node_attrs = graph.nodes(data=True)
    recognition = Recognition(intent=Intent("", confidence=1.0))

    # Step 1: go through path pairwise and gather input/output labels
    raw_sub_tokens: typing.List[typing.Tuple[str, str, typing.List[str]]] = []

    for last_node_tokens, next_node_tokens in pairwise(node_path):
        # Unpack path nodes
        if isinstance(last_node_tokens, int):
            last_node = last_node_tokens
            last_tokens = []
        else:
            last_node, last_tokens = last_node_tokens

        if isinstance(next_node_tokens, int):
            next_node = next_node_tokens
        else:
            next_node, _ = next_node_tokens

        # Get raw text
        word = node_attrs[next_node].get("word") or ""

        # Get output label
        edge_data = graph[last_node][next_node]
        olabel = edge_data.get("olabel") or ""

        if olabel == Word.WILDCARD:
            # Replace wildcard with actual tokens
            olabel = " ".join(last_tokens)

        if olabel[:10] == "__unpack__":
            # Decode payload as base64-encoded bytestring
            olabel = base64.decodebytes(olabel[10:].encode()).decode()

        if olabel[:9] == "__label__":
            # Intent name
            assert recognition.intent is not None
            recognition.intent.name = olabel[9:]
        elif word or olabel:
            # Keep non-empty words
            raw_sub_tokens.append((word, olabel, last_tokens))

    # Step 2: apply converters
    converter_stack: typing.List[ConverterInfo] = []
    raw_conv_tokens: typing.List[typing.Tuple[str, typing.Any, typing.List[str]]] = []

    for raw_token, sub_token, original_tokens in raw_sub_tokens:
        if sub_token and converter_stack and (sub_token[:2] != "__"):
            # Add to existing converter
            converter_stack[-1].tokens.append((raw_token, sub_token, original_tokens))
        elif sub_token[:11] == "__convert__":
            # Begin converter
            converter_key = sub_token[11:]
            converter_name = converter_key
            converter_args: typing.Optional[typing.List[str]] = None

            # Detect arguments
            if "," in converter_name:
                converter_name, *converter_args = converter_name.split(",")

            converter_stack.append(
                ConverterInfo(
                    key=converter_key, name=converter_name, args=converter_args
                )
            )
        elif sub_token[:13] == "__converted__":
            # End converter
            assert converter_stack, "Found __converted__ without a __convert__"
            last_converter = converter_stack.pop()
            actual_key = sub_token[13:]
            assert (
                last_converter.key == actual_key
            ), f"Mismatched converter name (expected {last_converter.key}, got {actual_key})"

            # Convert and add directly
            raw_tokens: typing.List[str] = []
            sub_tokens: typing.List[typing.Any] = []
            orig_tokens: typing.List[typing.List[str]] = []

            for t in last_converter.tokens:
                if t[0]:
                    raw_tokens.append(t[0])

                # Value may be 0
                if t[1] not in {None, ""}:
                    sub_tokens.append(t[1])

                if t[2]:
                    orig_tokens.append(t[2])

            # Run substituted tokens through conversion function
            converter_func = converters[last_converter.name]

            # Pass arguments as keyword "converter_args"
            converter_kwargs = (
                {"converter_args": last_converter.args} if last_converter.args else {}
            )
            converted_tokens = converter_func(*sub_tokens, **converter_kwargs)

            if converter_stack:
                # Add to parent converter
                target_list = converter_stack[-1].tokens
            else:
                # Add directly to list
                target_list = raw_conv_tokens

            # Zip 'em up
            target_list.extend(
                itertools.zip_longest(
                    raw_tokens, converted_tokens, orig_tokens, fillvalue=""
                )
            )
        else:
            raw_conv_tokens.append((raw_token, sub_token, original_tokens))

    assert not converter_stack, f"Converter(s) remaining on stack ({converter_stack})"

    # Step 3: collect entities
    entity_stack: typing.List[Entity] = []
    raw_index = 0
    sub_index = 0

    for raw_token, conv_token, original_tokens in raw_conv_tokens:
        # Handle raw (input) token
        if original_tokens:
            # Use tokens from recognition string
            recognition.raw_tokens.extend(original_tokens)
            raw_index += sum(len(t) for t in original_tokens) + len(original_tokens)

            if entity_stack:
                last_entity = entity_stack[-1]
                last_entity.raw_tokens.extend(original_tokens)
        elif raw_token:
            # Use word itself
            recognition.raw_tokens.append(raw_token)
            raw_index += len(raw_token) + 1

            if entity_stack:
                last_entity = entity_stack[-1]
                last_entity.raw_tokens.append(raw_token)

        # Handle converted (output) token
        if conv_token is not None:
            conv_token_str = str(conv_token)
            if conv_token_str:
                if conv_token_str[:9] == "__begin__":
                    # Begin tag/entity
                    entity_name = conv_token[9:]
                    entity_stack.append(
                        Entity(
                            entity=entity_name,
                            value="",
                            start=sub_index,
                            raw_start=raw_index,
                        )
                    )
                elif conv_token_str[:7] == "__end__":
                    # End tag/entity
                    assert entity_stack, "Found __end__ without a __begin__"
                    last_entity = entity_stack.pop()
                    actual_name = conv_token[7:]
                    assert (
                        last_entity.entity == actual_name
                    ), "Mismatched entity name (expected {last_entity.entity}, got {actual_name})"

                    # Assign end indexes
                    last_entity.end = sub_index - 1
                    last_entity.raw_end = raw_index - 1

                    # Create values
                    if len(last_entity.tokens) == 1:
                        # Use Python object
                        last_entity.value = last_entity.tokens[0]
                    else:
                        # Join as string
                        last_entity.value = " ".join(str(t) for t in last_entity.tokens)

                    last_entity.raw_value = " ".join(last_entity.raw_tokens)

                    # Add to recognition
                    recognition.entities.append(last_entity)
                elif conv_token_str[:10] == "__source__":
                    if entity_stack:
                        last_entity = entity_stack[-1]
                        last_entity.source = conv_token_str[10:]
                elif entity_stack:
                    # Add to most recent named entity
                    last_entity = entity_stack[-1]
                    last_entity.tokens.append(conv_token)

                    recognition.tokens.append(conv_token)
                    sub_index += len(conv_token_str) + 1
                else:
                    # Substituted text
                    recognition.tokens.append(conv_token)
                    sub_index += len(conv_token_str) + 1

    # Step 4: create text fields and compute confidence
    recognition.text = " ".join(str(t) for t in recognition.tokens)
    recognition.raw_text = " ".join(recognition.raw_tokens)

    if cost and cost > 0:
        # Set fuzzy confidence
        assert recognition.intent is not None

        recognition.intent.confidence = 1.0 - float(cost / len(recognition.raw_tokens))

    return RecognitionResult.SUCCESS, recognition


# -----------------------------------------------------------------------------


def bool_converter(obj: typing.Any) -> bool:
    """Returns False if argument is 'false' string or zero."""
    return (obj != 0) and (str(obj).lower() != "false")


def get_default_converters() -> typing.Dict[str, typing.Callable[..., typing.Any]]:
    """Get built-in fsticuffs converters"""
    return {
        "int": lambda *args: map(int, args),
        "float": lambda *args: map(float, args),
        "bool": lambda *args: map(bool_converter, args),
        "lower": lambda *args: map(str.lower, args),
        "upper": lambda *args: map(str.upper, args),
        "object": lambda *args: [
            {"value": args[0] if len(args) == 1 else " ".join(str(a) for a in args)}
        ],
        "kind": lambda *args, converter_args=None: [
            {"kind": converter_args[0], **a} for a in args
        ],
        "unit": lambda *args, converter_args=None: [
            {"unit": converter_args[0], **a} for a in args
        ],
        "num": lambda *args, converter_args=None: [
            extract_number(" ".join(str(a) for a in args), *(converter_args or []))
        ],
        "duration": lambda *args, converter_args=None: [
            extract_duration(" ".join(str(a) for a in args), *(converter_args or []))[0]
        ],
        "datetime": lambda *args, converter_args=None: [
            extract_datetime(" ".join(str(a) for a in args), *(converter_args or []))[0]
        ],
    }


# -----------------------------------------------------------------------------


def sample_by_intent(
    intent_graph: nx.DiGraph,
    num_samples: typing.Optional[int] = None,
    **recognition_args,
) -> typing.Dict[str, typing.List[Recognition]]:
    """Sample sentences from a graph."""

    sentences_by_intent: typing.Dict[str, typing.List[Recognition]] = defaultdict(list)

    start_node, end_node = get_start_end_nodes(intent_graph)
    assert (start_node is not None) and (
        end_node is not None
    ), "Missing start/end node(s)"

    if num_samples is not None:
        # Randomly sample
        paths = random.sample(
            list(nx.all_simple_paths(intent_graph, start_node, end_node)), num_samples
        )
    else:
        # Use generator
        paths = nx.all_simple_paths(intent_graph, start_node, end_node)

    for path in paths:
        _, recognition = path_to_recognition(path, intent_graph, **recognition_args)
        assert recognition, "Path failed"
        if recognition.intent:
            sentences_by_intent[recognition.intent.name].append(recognition)

    return sentences_by_intent
