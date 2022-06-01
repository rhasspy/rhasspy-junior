"""Parsing code for ini/JSGF grammars."""
import configparser
import io
import logging
import re
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .const import IntentsType, ReplacementsType, SentencesType
from .jsgf import (
    Expression,
    ParseMetadata,
    Rule,
    RuleReference,
    Sentence,
    Sequence,
    SequenceType,
    SlotReference,
    Word,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class Grammar:
    """Named JSGF grammar with rules."""

    grammar_name: str = ""
    rules: typing.List[Rule] = field(default_factory=list)

    GRAMMAR_DECLARATION = re.compile(r"^grammar ([^;]+);$")

    @classmethod
    def parse(cls, source: typing.TextIO) -> "Grammar":
        """Parse single JSGF grammar."""
        grammar = Grammar()

        # Read line-by-line
        for line in source:
            line = line.strip()
            if line[:1] == "#" or (not line):
                # Skip comments/blank lines
                continue

            grammar_match = Grammar.GRAMMAR_DECLARATION.match(line)
            if grammar_match is not None:
                # grammar GrammarName;
                grammar.grammar_name = grammar_match.group(1)
            else:
                # public <RuleName> = rule body;
                # <RuleName> = rule body;
                grammar.rules.append(Rule.parse(line))  # pylint: disable=E1101

        return grammar


# -----------------------------------------------------------------------------


def parse_ini(
    source: typing.Union[str, Path, typing.TextIO],
    intent_filter: typing.Optional[typing.Callable[[str], bool]] = None,
    sentence_transform: typing.Callable[[str], str] = None,
    file_name: typing.Optional[str] = None,
) -> IntentsType:
    """Parse multiple JSGF grammars from an ini file."""
    intent_filter = intent_filter or (lambda x: True)
    if isinstance(source, str):
        source = io.StringIO(source)
        file_name = file_name or "<StringIO>"
    elif isinstance(source, Path):
        # pylint: disable=consider-using-with
        source = open(source, "r", encoding="utf-8")
        file_name = file_name or str(source)
    else:
        file_name = file_name or "<TextIO>"

    # Process configuration sections
    sentences: IntentsType = defaultdict(list)

    try:
        # Create ini parser
        config = configparser.ConfigParser(
            allow_no_value=True, strict=False, delimiters=["="]
        )

        # case sensitive
        config.optionxform = str  # type: ignore
        config.read_file(source)

        _LOGGER.debug("Loaded ini file")

        # Parse each section (intent)
        line_number: int = 1
        for sec_name in config.sections():
            # Exclude if filtered out.
            if not intent_filter(sec_name):
                _LOGGER.debug("Skipping %s", sec_name)
                continue

            # Section header
            line_number += 1

            # Processs settings (sentences/rules)
            for k, v in config[sec_name].items():
                if v is None:
                    # Collect non-valued keys as sentences
                    sentence = k.strip()

                    # Fix \[ escape sequence
                    sentence = sentence.replace("\\[", "[")

                    if sentence_transform:
                        # Do transform
                        sentence = sentence_transform(sentence)

                    sentences[sec_name].append(
                        Sentence.parse(
                            sentence,
                            metadata=ParseMetadata(
                                file_name=file_name,
                                line_number=line_number,
                                intent_name=sec_name,
                            ),
                        )
                    )
                else:
                    sentence = v.strip()

                    if sentence_transform:
                        # Do transform
                        sentence = sentence_transform(sentence)

                    # Collect key/value pairs as JSGF rules
                    rule = f"<{k.strip()}> = ({sentence});"

                    # Fix \[ escape sequence
                    rule = rule.replace("\\[", "[")

                    sentences[sec_name].append(
                        Rule.parse(
                            rule,
                            metadata=ParseMetadata(
                                file_name=file_name,
                                line_number=line_number,
                                intent_name=sec_name,
                            ),
                        )
                    )

                # Sentence
                line_number += 1

            # Blank line
            line_number += 1
    finally:
        source.close()

    return sentences


# -----------------------------------------------------------------------------


def split_rules(
    intents: IntentsType, replacements: typing.Optional[ReplacementsType] = None
) -> typing.Tuple[SentencesType, ReplacementsType]:
    """Seperate out rules and sentences from all intents."""
    sentences: SentencesType = {}
    replacements = replacements or {}

    for intent_name, intent_exprs in intents.items():
        sentences[intent_name] = []

        # Extract rules and fold them into replacements
        for expr in intent_exprs:
            if isinstance(expr, Rule):
                # Rule
                rule_name = expr.rule_name

                # Surround with <>
                rule_name = f"<{intent_name}.{rule_name}>"
                replacements[rule_name] = [expr.rule_body]
            else:
                sentences[intent_name].append(expr)

    return sentences, replacements


# -----------------------------------------------------------------------------


def get_intent_counts(
    sentences: SentencesType,
    replacements: typing.Optional[ReplacementsType] = None,
    exclude_slots: bool = True,
    count_dict: typing.Optional[typing.Dict[Expression, int]] = None,
) -> typing.Dict[str, int]:
    """Get number of possible sentences for each intent."""
    intent_counts: typing.Dict[str, int] = defaultdict(int)

    for intent_name, intent_sentences in sentences.items():
        # Compute counts for all sentences
        intent_counts[intent_name] = max(
            1,
            sum(
                get_expression_count(
                    s, replacements, exclude_slots=exclude_slots, count_dict=count_dict
                )
                for s in intent_sentences
            ),
        )

    return intent_counts


# -----------------------------------------------------------------------------


def get_expression_count(
    expression: Expression,
    replacements: typing.Optional[ReplacementsType] = None,
    exclude_slots: bool = True,
    count_dict: typing.Optional[typing.Dict[Expression, int]] = None,
) -> int:
    """Get the number of possible sentences in an expression."""
    if isinstance(expression, Sequence):
        if expression.type == SequenceType.GROUP:
            # Counts multiply down the sequence
            count = 1
            for sub_item in expression.items:
                count = count * get_expression_count(
                    sub_item,
                    replacements,
                    exclude_slots=exclude_slots,
                    count_dict=count_dict,
                )

            if count_dict is not None:
                count_dict[expression] = count

            return count

        if expression.type == SequenceType.ALTERNATIVE:
            # Counts sum across the alternatives
            count = sum(
                get_expression_count(
                    sub_item,
                    replacements,
                    exclude_slots=exclude_slots,
                    count_dict=count_dict,
                )
                for sub_item in expression.items
            )

            if count_dict is not None:
                count_dict[expression] = count

            return count
    elif isinstance(expression, RuleReference):
        # Get substituted sentences for <rule>
        key = f"<{expression.full_rule_name}>"
        assert replacements, key
        count = sum(
            get_expression_count(
                value, replacements, exclude_slots=exclude_slots, count_dict=count_dict
            )
            for value in replacements[key]
        )

        if count_dict is not None:
            count_dict[expression] = count

        return count
    elif (not exclude_slots) and isinstance(expression, SlotReference):
        # Get substituted sentences for $slot
        key = f"${expression.slot_name}"
        assert replacements, key
        count = sum(
            get_expression_count(
                value, replacements, exclude_slots=exclude_slots, count_dict=count_dict
            )
            for value in replacements[key]
        )

        if count_dict is not None:
            count_dict[expression] = count

        return count
    elif isinstance(expression, Word):
        # Single word
        count = 1
        if count_dict is not None:
            count_dict[expression] = count

        return count

    # Unknown expression type
    count = 0
    if count_dict is not None:
        count_dict[expression] = count

    return count
