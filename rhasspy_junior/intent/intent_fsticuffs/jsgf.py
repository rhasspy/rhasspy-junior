"""Parses a subset of JSGF into objects."""
import re
import typing
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Substitutable:
    """Indicates an expression may be replaced with some text."""

    # Replacement text
    substitution: typing.Optional[typing.Union[str, typing.List[str]]] = None

    # Names of converters to apply after substitution
    converters: typing.List[str] = field(default_factory=list)

    @staticmethod
    def parse_substitution(sub_text: str) -> typing.Union[str, typing.List[str]]:
        """Parse substitution text into token list or string."""
        sub_text = sub_text.strip()

        if sub_text[:1] == "(":
            sub_text = sub_text[1:]

        if sub_text[-1:] == ")":
            sub_text = sub_text[:-1]

        if " " in sub_text:
            return sub_text.split()

        return sub_text


@dataclass
class Tag(Substitutable):
    """{tag} attached to an expression."""

    # Name of tag (entity)
    tag_text: str = ""


@dataclass
class Taggable:
    """Indicates an expression may be tagged."""

    # Tag to be applied
    tag: typing.Optional[Tag] = None


@dataclass
class Expression:
    """Base class for most JSGF types."""

    # Text representation expression
    text: str = ""


@dataclass
class Word(Substitutable, Taggable, Expression):
    """Single word/token."""

    WILDCARD = "*"

    @property
    def is_wildcard(self):
        """True if this word is a wildcard"""
        return self.text == Word.WILDCARD


class SequenceType(str, Enum):
    """Type of a sequence. Optionals are alternatives with an empty option."""

    # Sequence of expressions
    GROUP = "group"

    # Expressions where only one will be recognized
    ALTERNATIVE = "alternative"


@dataclass
class Sequence(Substitutable, Taggable, Expression):
    """Ordered sequence of expressions. Supports groups, optionals, and alternatives."""

    # Items in the sequence
    items: typing.List[Expression] = field(default_factory=list)

    # Group or alternative
    type: SequenceType = SequenceType.GROUP


@dataclass
class RuleReference(Taggable, Expression):
    """Reference to a rule by <name> or <grammar.name>."""

    # Name of referenced rule
    rule_name: str = ""

    # Grammar name of referenced rule
    grammar_name: typing.Optional[str] = None

    @property
    def full_rule_name(self):
        """Get fully qualified rule name."""
        if self.grammar_name:
            return f"{self.grammar_name}.{self.rule_name}"

        return self.rule_name


@dataclass
class SlotReference(Substitutable, Taggable, Expression):
    """Reference to a slot by $name."""

    # Name of referenced slot
    slot_name: str = ""


@dataclass
class ParseMetadata:
    """Debug metadata for more helpful parsing errors."""

    file_name: str
    line_number: int
    intent_name: typing.Optional[str] = None


@dataclass
class Sentence(Sequence):
    """Sequence representing a complete sentence template."""

    @staticmethod
    def parse(text: str, metadata: typing.Optional[ParseMetadata] = None) -> "Sentence":
        """Parse a single sentence."""
        s = Sentence(text=text)
        parse_expression(s, text, metadata=metadata)
        return Sentence(
            text=s.text,
            items=s.items,
            type=s.type,
            tag=s.tag,
            substitution=s.substitution,
        )


@dataclass
class Rule:
    """Named rule with body."""

    RULE_DEFINITION = re.compile(r"^(public)?\s*<([^>]+)>\s*=\s*([^;]+)(;)?$")

    rule_name: str
    rule_body: Sentence
    public: bool = False
    text: str = ""

    @staticmethod
    def parse(text: str, metadata: typing.Optional[ParseMetadata] = None) -> "Rule":
        """Parse a single rule."""
        # public <RuleName> = rule body;
        # <RuleName> = rule body;
        rule_match = Rule.RULE_DEFINITION.match(text)
        assert rule_match is not None, f"No rule was found in {text}"

        public = rule_match.group(1) is not None
        rule_name = rule_match.group(2)
        rule_text = rule_match.group(3)

        s = Sentence.parse(rule_text, metadata=metadata)
        return Rule(rule_name=rule_name, rule_body=s, public=public, text=text)


# -----------------------------------------------------------------------------


def walk_expression(
    expression: Expression,
    visit: typing.Callable[
        [Expression], typing.Union[bool, typing.Optional[Expression]]
    ],
    replacements: typing.Optional[typing.Dict[str, typing.List[Expression]]] = None,
) -> typing.Union[bool, typing.Optional[Expression]]:
    """Recursively visit/replace nodes in expression."""
    result = visit(expression)

    if result is False:
        return False

    if result is not None:
        assert isinstance(result, Expression), f"Expected Expression, got {result}"
        expression = result

    if isinstance(expression, Sequence):
        # pylint: disable=consider-using-enumerate
        for i in range(len(expression.items)):
            new_item = walk_expression(expression.items[i], visit, replacements)
            if new_item:
                assert isinstance(
                    new_item, Expression
                ), f"Expected Expression, got {new_item}"
                expression.items[i] = new_item
    elif isinstance(expression, Rule):
        new_body = walk_expression(expression.rule_body, visit, replacements)
        if new_body:
            assert isinstance(new_body, Sentence), f"Expected Sentence, got {new_body}"
            expression.rule_body = new_body
    elif isinstance(expression, RuleReference):
        key = f"<{expression.full_rule_name}>"
        if replacements and (key in replacements):
            key_replacements = replacements[key]

            # pylint: disable=consider-using-enumerate
            for i in range(len(key_replacements)):
                new_item = walk_expression(key_replacements[i], visit, replacements)
                if new_item:
                    assert isinstance(
                        new_item, Expression
                    ), f"Expected Expression, got {new_item}"
                    key_replacements[i] = new_item
    elif isinstance(expression, SlotReference):
        key = f"${expression.slot_name}"
        if replacements and (key in replacements):
            key_replacements = replacements[key]

            # pylint: disable=consider-using-enumerate
            for i in range(len(key_replacements)):
                new_item = walk_expression(key_replacements[i], visit, replacements)
                if new_item:
                    assert isinstance(
                        new_item, Expression
                    ), f"Expected Expression, got {new_item}"
                    key_replacements[i] = new_item

    return expression


# -----------------------------------------------------------------------------


def maybe_remove_parens(s: str) -> str:
    """Remove parentheses from around a string if it has them."""
    if (len(s) > 1) and (s[0] == "(") and (s[-1] == ")"):
        return s[1:-1]

    return s


def split_words(text: str) -> typing.Iterable[Expression]:
    """Split words by whitespace. Detect slot references and substitutions."""
    tokens: typing.List[str] = []
    token: str = ""
    last_c: str = ""
    in_seq_sub: bool = False

    # Process words, correctly handling substitution sequences.
    # e.g., input:(output words)
    for c in text:
        break_token = False

        if (c == "(") and (last_c == ":"):
            # Begin sequence substitution
            in_seq_sub = True
        elif in_seq_sub and (c == ")"):
            # End sequence substitution
            in_seq_sub = False
            break_token = True
        elif c == " " and (not in_seq_sub):
            # Whitespace break (not inside sequence substitution)
            break_token = True
        else:
            # Accumulate into token
            token += c

        if break_token and token:
            tokens.append(token)
            token = ""

        last_c = c

    if token:
        # Last token
        tokens.append(token)

    for token in tokens:
        if token[:1] == "$":
            slot_name = token[1:]
            if ":" in slot_name:
                # Slot with substitutions
                slot_name, substitution = slot_name.split(":", maxsplit=1)
                yield SlotReference(
                    text=token,
                    slot_name=slot_name,
                    substitution=Substitutable.parse_substitution(substitution),
                )
            else:
                # Slot without substitutions
                yield SlotReference(text=token, slot_name=slot_name)
        else:
            word = Word(text=token)

            if "!" in token:
                # Word with converter(s)
                # e.g., twenty:20!int
                parts = token.split("!")
                word.text = parts[0]
                word.converters = parts[1:]

            if ":" in word.text:
                # Word with substitution
                # e.g., twenty:20
                lhs, rhs = word.text.split(":", maxsplit=1)
                word.text = lhs
                word.substitution = Substitutable.parse_substitution(rhs)

            yield word


def parse_expression(
    root: typing.Optional[Sequence],
    text: str,
    end: typing.List[str] = None,
    is_literal: bool = True,
    metadata: typing.Optional[ParseMetadata] = None,
) -> typing.Optional[int]:
    """Parse a full expression. Return index in text where current expression ends."""
    end = end or []
    found: bool = False
    next_index: int = 0
    literal: str = ""
    last_taggable: typing.Optional[Taggable] = None
    last_group: typing.Optional[Sequence] = root
    escape_depth: int = 0

    # Process text character-by-character
    for current_index, c in enumerate(text):
        if current_index < next_index:
            # Skip ahread
            current_index += 1
            continue

        # Get previous character
        if current_index > 0:
            last_c = text[current_index - 1]
        else:
            last_c = ""

        # Handle escaped characters (e.g., "\\" and "\(")
        if escape_depth > 0:
            literal += c
            escape_depth -= 1
            continue

        if c == "\\":
            escape_depth += 1
            continue

        next_index = current_index + 1

        if c in end:
            # Found end character of expression (e.g., ])
            next_index += 1
            found = True
            break

        if (c in {":", "!"}) and (last_c in {")", "]"}):
            # Handle sequence substitution/conversion
            assert isinstance(last_taggable, Substitutable), parse_error(
                f"Expected Substitutable, got {last_taggable}",
                text,
                current_index,
                metadata=metadata,
            )

            next_seq_sub = False

            if next_index < len(text):
                # Check for substitution sequence.
                # e.g., (input words):(output words)
                if text[next_index] == "(":
                    # Find end of group
                    next_end = [")"] + end
                    next_seq_sub = True
                else:
                    # Find end of word
                    next_end = [" "] + end

                next_index = parse_expression(
                    None,
                    text[current_index + 1 :],
                    next_end,
                    is_literal=False,
                    metadata=metadata,
                )

                if next_index is None:
                    # End of text
                    next_index = len(text) + 1
                else:
                    next_index += current_index - 1
            else:
                # End of text
                next_index = len(text) + 1

            if next_seq_sub:
                # Consume end paren
                next_index += 1
                assert text[next_index - 1] == ")", parse_error(
                    "Missing end parenthesis", text, current_index, metadata=metadata
                )

            if c == ":":
                # Substitution/conversion
                sub_text = text[current_index + 1 : next_index].strip()

                if "!" in sub_text:
                    # Extract converter(s)
                    sub_text, *converters = sub_text.split("!")
                    last_taggable.converters = converters

                last_taggable.substitution = Substitutable.parse_substitution(sub_text)
            else:
                # Conversion only
                conv_text = maybe_remove_parens(
                    text[current_index + 1 : next_index].strip()
                )
                last_taggable.converters = conv_text.split("!")

        elif (c == "(" and last_c != ":") or (c in {"<", "[", "{", "|"}):
            # Begin group/tag/alt/etc.

            # Break literal here
            literal = literal.strip()
            if literal:
                assert last_group is not None, parse_error(
                    "No group preceeding literal",
                    text,
                    current_index,
                    metadata=metadata,
                )
                words = list(split_words(literal))
                last_group.items.extend(words)

                last_word = words[-1]
                assert isinstance(last_word, Taggable), parse_error(
                    f"Expected Taggable, got {last_word}",
                    text,
                    current_index,
                    metadata=metadata,
                )
                last_taggable = last_word
                literal = ""

            if c == "<":
                # Rule reference
                assert last_group is not None, parse_error(
                    "No group preceeding rule reference",
                    text,
                    current_index,
                    metadata=metadata,
                )
                rule = RuleReference()
                end_index = parse_expression(
                    None,
                    text[current_index + 1 :],
                    end=[">"],
                    is_literal=False,
                    metadata=metadata,
                )
                assert end_index, parse_error(
                    "Failed to find ending '>'", text, current_index, metadata=metadata
                )
                next_index = end_index + current_index

                rule_name = text[current_index + 1 : next_index - 1]
                if "." in rule_name:
                    # Split by last dot
                    last_dot = rule_name.rindex(".")
                    rule.grammar_name = rule_name[:last_dot]
                    rule.rule_name = rule_name[last_dot + 1 :]
                else:
                    # Use entire name
                    rule.rule_name = rule_name

                    if metadata:
                        # Use intent name for grammar name
                        rule.grammar_name = metadata.intent_name

                rule.text = text[current_index:next_index]
                last_group.items.append(rule)
                last_taggable = rule
            elif c == "(":
                # Group (expression)
                if last_group is not None:
                    # Parse group into sequence.
                    # If last_group is None, we're on the right-hand side of a
                    # ":" and the text will be interpreted as a substitution
                    # instead.
                    group = Sequence(type=SequenceType.GROUP)
                    end_index = parse_expression(
                        group, text[current_index + 1 :], end=[")"], metadata=metadata
                    )
                    assert end_index, parse_error(
                        "Failed to find ending ')'",
                        text,
                        current_index,
                        metadata=metadata,
                    )
                    next_index = end_index + current_index

                    group.text = remove_escapes(
                        text[current_index + 1 : next_index - 1]
                    )
                    last_group.items.append(group)
                    last_taggable = group
            elif c == "[":
                # Optional
                # Recurse with group sequence to capture multi-word children.
                optional_seq = Sequence(type=SequenceType.GROUP)
                end_index = parse_expression(
                    optional_seq,
                    text[current_index + 1 :],
                    end=["]"],
                    metadata=metadata,
                )
                assert end_index, parse_error(
                    "Failed to find ending ']'", text, current_index, metadata=metadata
                )
                next_index = end_index + current_index

                optional = Sequence(type=SequenceType.ALTERNATIVE)
                if optional_seq.items:
                    if (
                        (len(optional_seq.items) == 1)
                        and (not optional_seq.tag)
                        and (not optional_seq.substitution)
                    ):
                        # Unpack inner item
                        inner_item = optional_seq.items[0]

                        optional.items.append(inner_item)
                    elif optional_seq.type == SequenceType.ALTERNATIVE:
                        # Unwrap inner alternative
                        optional.items.extend(optional_seq.items)
                    else:
                        # Keep inner group
                        optional_seq.text = remove_escapes(
                            text[current_index + 1 : next_index - 1]
                        )

                        optional.items.append(optional_seq)

                # Empty alternative
                optional.items.append(Word(text=""))
                optional.text = remove_escapes(text[current_index + 1 : next_index - 1])

                assert last_group is not None, parse_error(
                    "Expected group preceeding optional",
                    text,
                    current_index,
                    metadata=metadata,
                )
                last_group.items.append(optional)
                last_taggable = optional
            elif c == "{":
                assert last_taggable is not None, parse_error(
                    "Expected expression preceeding tag",
                    text,
                    current_index,
                    metadata=metadata,
                )
                tag = Tag()

                # Tag
                end_index = parse_expression(
                    None,
                    text[current_index + 1 :],
                    end=["}"],
                    is_literal=False,
                    metadata=metadata,
                )
                assert end_index, parse_error(
                    "Failed to find ending '}}'",
                    text,
                    current_index,
                    metadata=metadata,
                )
                next_index = end_index + current_index

                # Exclude {}
                tag.tag_text = remove_escapes(text[current_index + 1 : next_index - 1])

                # Handle substitution/converter(s)
                if "!" in tag.tag_text:
                    # Word with converter(s)
                    # e.g., twenty:20!int
                    parts = tag.tag_text.split("!")
                    tag.tag_text = parts[0]
                    tag.converters = parts[1:]

                if ":" in tag.tag_text:
                    # Word with substitution
                    # e.g., twenty:20
                    lhs, rhs = tag.tag_text.split(":", maxsplit=1)
                    tag.tag_text = lhs
                    tag.substitution = Substitutable.parse_substitution(rhs)

                last_taggable.tag = tag
            elif c == "|":
                assert root is not None, parse_error(
                    "Unexpected '|' outside of group/alternative",
                    text,
                    current_index,
                    metadata=metadata,
                )
                if root.type != SequenceType.ALTERNATIVE:
                    # Create alternative
                    alternative = Sequence(type=SequenceType.ALTERNATIVE)
                    if len(root.items) == 1:
                        # Add directly
                        alternative.items.append(root.items[0])
                    else:
                        # Wrap in group
                        last_group = Sequence(type=SequenceType.GROUP, items=root.items)
                        alternative.items.append(last_group)

                    # Modify original sequence
                    root.items = [alternative]

                    # Overwrite root
                    root = alternative

                assert last_group is not None, parse_error(
                    "Expected group preceeding alternative",
                    text,
                    current_index,
                    metadata=metadata,
                )
                if not last_group.text:
                    # Fix text
                    last_group.text = " ".join(item.text for item in last_group.items)

                # Create new group for any follow-on expressions
                last_group = Sequence(type=SequenceType.GROUP)

                alternative.items.append(last_group)
        else:
            # Accumulate into current literal
            literal += c

    # End of expression
    current_index = len(text)

    # Break literal
    literal = literal.strip()
    if is_literal and literal:
        assert root is not None, parse_error(
            "Literal outside parent expression", text, current_index, metadata=metadata
        )
        words = list(split_words(literal))
        assert last_group is not None, parse_error(
            "Expected group preceeding literal", text, current_index, metadata=metadata
        )
        last_group.items.extend(words)

    if last_group:
        if not last_group.text:
            # Fix text
            last_group.text = " ".join(item.text for item in last_group.items)

        if len(last_group.items) == 1:
            # Simplify final group
            assert root is not None, parse_error(
                "Group outside parent expression",
                text,
                current_index,
                metadata=metadata,
            )
            root.items[-1] = last_group.items[0]

            # Force text to be fixed
            root.text = ""

    if root and (not root.text):
        # Fix text
        if root.type == SequenceType.ALTERNATIVE:
            # Pipe separated
            root.text = " | ".join(item.text for item in root.items)
        else:
            # Space separated
            root.text = " ".join(item.text for item in root.items)

    if end and (not found):
        # Signal end not found
        return None

    return next_index


def parse_error(
    error: str, text: str, column: int, metadata: typing.Optional[ParseMetadata] = None
) -> str:
    """Generate helpful parsing error if metadata is available."""
    if metadata:
        return f"{error} (text='{text}', file={metadata.file_name}, column={column}, line={metadata.line_number})"

    return error


def remove_escapes(text: str) -> str:
    """Remove backslash escape sequences"""
    return re.sub(r"\\(.)", r"\1", text)
