"""Methods for generating ASR artifacts."""
import io
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

import networkx as nx

from .g2p import PronunciationsType, write_pronunciations

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)

DEFAULT_SCOPE = ""

# -----------------------------------------------------------------------------


def train(
    graph: nx.DiGraph,
    pronunciations: PronunciationsType,
    model_dir: typing.Union[str, Path],
    graph_dir: typing.Union[str, Path],
    kaldi_dir: typing.Union[str, Path],
    dictionary: typing.Optional[typing.Union[str, Path]] = None,
    language_model: typing.Optional[typing.Union[str, Path]] = None,
    dictionary_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
    g2p_model: typing.Optional[typing.Union[str, Path]] = None,
    g2p_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
    missing_words_path: typing.Optional[Path] = None,
    vocab_path: typing.Optional[typing.Union[str, Path]] = None,
    balance_counts: bool = True,
    spn_phone: str = "SPN",
    sil_phone: str = "SIL",
    eps: str = "<eps>",
    frequent_words: typing.Optional[typing.Set[str]] = None,
    sil: str = "<sil>",
    unk: str = "<unk>",
):
    """Re-generates HCLG.fst from intent graph"""
    g2p_word_transform = g2p_word_transform or (lambda s: s)
    model_dir = Path(model_dir)

    vocabulary: typing.Set[str] = set()

    if vocab_path:
        vocab_file = open(vocab_path, "w+", encoding="utf-8")
    else:
        vocab_file = typing.cast(
            # pylint: disable=consider-using-with
            io.TextIOWrapper,
            tempfile.NamedTemporaryFile(suffix=".txt", mode="w+"),
        )
        vocab_path = vocab_file.name

    intent_state_path = model_dir / "intent_states.txt"
    intent_state_path.parent.mkdir(parents=True, exist_ok=True)

    # pylint: disable=consider-using-with
    intent_state_file = open(intent_state_path, "w", encoding="utf-8")

    # Begin training
    with tempfile.NamedTemporaryFile(mode="w+") as lm_file:
        with vocab_file:
            _LOGGER.debug("Writing G.fst directly")
            graph_to_g_fst(
                graph,
                lm_file,
                vocab_file,
                intent_state_file=intent_state_file,
                eps=eps,
            )

            # Load vocabulary
            vocab_file.seek(0)
            vocabulary.update(line.strip() for line in vocab_file)

        assert vocabulary, "No words in vocabulary"

        # <unk> - unknown word
        vocabulary.add(unk)
        pronunciations[unk] = [[spn_phone]]

        # <sil> - silence
        vocabulary.add(sil)
        pronunciations[sil] = [[sil_phone]]

        # Write dictionary to temporary file
        with tempfile.NamedTemporaryFile(mode="w+") as dictionary_file:
            _LOGGER.debug("Writing pronunciation dictionary")
            write_pronunciations(
                vocabulary,
                pronunciations,
                dictionary_file.name,
                g2p_model=g2p_model,
                g2p_word_transform=g2p_word_transform,
                missing_words_path=missing_words_path,
                sil_phone=sil_phone,
            )

            # -----------------------------------------------------------------

            dictionary_file.seek(0)
            if dictionary:
                # Copy dictionary over real file
                shutil.copy(dictionary_file.name, dictionary)
                _LOGGER.debug("Wrote dictionary to %s", str(dictionary))
            else:
                dictionary = Path(dictionary_file.name)
                dictionary_file.seek(0)

            lm_file.seek(0)
            if language_model:
                # Copy language model over real file
                shutil.copy(lm_file.name, language_model)
                _LOGGER.debug("Wrote language model to %s", str(language_model))
            else:
                language_model = Path(lm_file.name)
                lm_file.seek(0)

            # Generate HCLG.fst
            train_kaldi(
                model_dir,
                graph_dir,
                dictionary,
                language_model,
                kaldi_dir=kaldi_dir,
                eps=eps,
                sil=sil,
                unk=unk,
            )


# -----------------------------------------------------------------------------


def graph_to_g_fst(
    graph: nx.DiGraph,
    fst_file: typing.IO[str],
    vocab_file: typing.IO[str],
    intent_state_file: typing.Optional[typing.IO[str]] = None,
    eps: str = "<eps>",
):
    """
    Write G.fst text file using intent graph.

    Compiled later on with fstcompile.
    """
    vocabulary: typing.Set[str] = set()

    n_data = graph.nodes(data=True)
    final_states: typing.Set[int] = set()
    state_map: typing.Dict[int, int] = {}

    # start state
    start_node: int = next(n for n, data in n_data if data.get("start"))

    # Transitions
    for _, intent_node, intent_edge_data in graph.edges(start_node, data=True):
        # Map states starting from 0
        from_state = state_map.get(start_node, len(state_map))
        state_map[start_node] = from_state

        to_state = state_map.get(intent_node, len(state_map))
        state_map[intent_node] = to_state

        print(from_state, to_state, eps, eps, 0.0, file=fst_file)

        if intent_state_file is not None:
            # Write <from> <to> <intent>
            intent_name = intent_edge_data["olabel"][len("__label__") :]
            print(from_state, to_state, intent_name, file=intent_state_file)

        # Add intent sub-graphs
        for edge in nx.edge_bfs(graph, intent_node):
            edge_data = graph.edges[edge]
            from_node, to_node = edge

            # Get input/output labels.
            # Empty string indicates epsilon transition (eps)
            ilabel = edge_data.get("ilabel", "") or eps

            # Check for whitespace
            assert (
                " " not in ilabel
            ), f"Input symbol cannot contain whitespace: {ilabel}"

            if ilabel != eps:
                vocabulary.add(ilabel)

            # Map states starting from 0
            from_state = state_map.get(from_node, len(state_map))
            state_map[from_node] = from_state

            to_state = state_map.get(to_node, len(state_map))
            state_map[to_node] = to_state

            print(from_state, to_state, ilabel, ilabel, 0.0, file=fst_file)

            # Check if final state
            is_from_final = n_data[from_node].get("final", False)
            is_to_final = n_data[to_node].get("final", False)

            if is_from_final:
                final_states.add(from_state)

            if is_to_final:
                final_states.add(to_state)

    # Record final states
    for final_state in final_states:
        print(final_state, 0.0, file=fst_file)

    # Write vocabulary
    for word in vocabulary:
        print(word, file=vocab_file)


# -----------------------------------------------------------------------------


def train_kaldi(
    model_dir: typing.Union[str, Path],
    graph_dir: typing.Union[str, Path],
    dictionary: typing.Union[str, Path],
    language_model: typing.Union[str, Path],
    kaldi_dir: typing.Union[str, Path],
    eps: str = "<eps>",
    sil: str = "<sil>",
    unk: str = "<unk>",
):
    """Generates HCLG.fst from dictionary and language model."""

    # Convert to paths
    model_dir = Path(model_dir)
    graph_dir = Path(graph_dir)
    kaldi_dir = Path(kaldi_dir)

    # -------------------------------------------------------------------------
    # Kaldi Training
    # ---------------------------------------------------------
    # 1. prepare_lang.sh
    # 2. format_lm.sh (or fstcompile)
    # 3. mkgraph.sh
    # 4. prepare_online_decoding.sh
    # ---------------------------------------------------------

    # Extend PATH
    egs_utils_dir = kaldi_dir / "egs" / "wsj" / "s5" / "utils"
    extended_env = os.environ.copy()
    extended_env["PATH"] = (
        str(kaldi_dir) + ":" + str(egs_utils_dir) + ":" + extended_env["PATH"]
    )

    # Create empty path.sh
    path_sh = model_dir / "path.sh"
    if not path_sh.is_file():
        path_sh.write_text("")

    # Delete existing data/graph
    data_dir = model_dir / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    if graph_dir.exists():
        shutil.rmtree(graph_dir)

    data_local_dir = model_dir / "data" / "local"

    _LOGGER.debug("Generating lexicon")
    dict_local_dir = data_local_dir / "dict"
    dict_local_dir.mkdir(parents=True, exist_ok=True)

    # Copy phones
    phones_dir = model_dir / "phones"
    for phone_file in phones_dir.glob("*.txt"):
        shutil.copy(phone_file, dict_local_dir / phone_file.name)

    # Copy dictionary
    shutil.copy(dictionary, dict_local_dir / "lexicon.txt")

    # Create utils link
    model_utils_link = model_dir / "utils"

    try:
        # Can't use missing_ok in 3.6
        model_utils_link.unlink()
    except Exception:
        pass

    model_utils_link.symlink_to(egs_utils_dir, target_is_directory=True)

    # 1. prepare_lang.sh
    lang_dir = data_dir / "lang"
    lang_local_dir = data_local_dir / "lang"
    prepare_lang = [
        "bash",
        str(egs_utils_dir / "prepare_lang.sh"),
        str(dict_local_dir),
        unk,
        str(lang_local_dir),
        str(lang_dir),
    ]

    _LOGGER.debug(prepare_lang)
    subprocess.check_call(prepare_lang, cwd=model_dir, env=extended_env)

    # 2. fstcompile > G.fst
    compile_grammar = [
        "fstcompile",
        shlex.quote(f"--isymbols={lang_dir}/words.txt"),
        shlex.quote(f"--osymbols={lang_dir}/words.txt"),
        "--keep_isymbols=false",
        "--keep_osymbols=false",
        shlex.quote(str(language_model)),
        shlex.quote(str(lang_dir / "G.fst.unsorted")),
    ]

    _LOGGER.debug(compile_grammar)
    subprocess.check_call(compile_grammar, cwd=model_dir, env=extended_env)

    arcsort = [
        "fstarcsort",
        "--sort_type=ilabel",
        shlex.quote(str(lang_dir / "G.fst.unsorted")),
        shlex.quote(str(lang_dir / "G.fst")),
    ]

    _LOGGER.debug(arcsort)
    subprocess.check_call(arcsort, cwd=model_dir, env=extended_env)
    os.unlink(lang_dir / "G.fst.unsorted")

    # 3. mkgraph.sh
    mkgraph = [
        "bash",
        str(egs_utils_dir / "mkgraph.sh"),
        "--self-loop-scale",
        "1.0",
        str(lang_dir),
        str(model_dir / "model"),
        str(graph_dir),
    ]
    _LOGGER.debug(mkgraph)
    subprocess.check_call(mkgraph, cwd=model_dir, env=extended_env)

    # 4. prepare_online_decoding.sh
    train_prepare_online_decoding(model_dir, lang_dir, kaldi_dir)


def train_prepare_online_decoding(
    model_dir: typing.Union[str, Path],
    lang_dir: typing.Union[str, Path],
    kaldi_dir: typing.Union[str, Path],
):
    """Prepare model for online decoding."""
    model_dir = Path(model_dir)
    kaldi_dir = Path(kaldi_dir)

    # prepare_online_decoding.sh (nnet3 only)
    extractor_dir = model_dir / "extractor"
    if extractor_dir.is_dir():
        # Extend PATH
        egs_utils_dir = kaldi_dir / "egs" / "wsj" / "s5" / "utils"
        extended_env = os.environ.copy()
        extended_env["PATH"] = (
            str(kaldi_dir) + ":" + str(egs_utils_dir) + ":" + extended_env["PATH"]
        )

        # Create empty path.sh
        path_sh = model_dir / "path.sh"
        if not path_sh.is_file():
            path_sh.write_text("")

        # Create utils link
        model_utils_link = model_dir / "utils"

        try:
            # Can't use missing_ok in 3.6
            model_utils_link.unlink()
        except Exception:
            pass

        model_utils_link.symlink_to(egs_utils_dir, target_is_directory=True)

        # Generate online.conf
        mfcc_conf = model_dir / "conf" / "mfcc_hires.conf"
        egs_steps_dir = kaldi_dir / "egs" / "wsj" / "s5" / "steps"
        prepare_online_decoding = [
            "bash",
            str(egs_steps_dir / "online" / "nnet3" / "prepare_online_decoding.sh"),
            "--mfcc-config",
            str(mfcc_conf),
            str(lang_dir),
            str(extractor_dir),
            str(model_dir / "model"),
            str(model_dir / "online"),
        ]

        _LOGGER.debug(prepare_online_decoding)
        subprocess.run(
            prepare_online_decoding,
            cwd=model_dir,
            env=extended_env,
            stderr=subprocess.STDOUT,
            check=True,
        )


def prune_intents(
    model_dir: typing.Union[str, Path],
    graph_dir: typing.Union[str, Path],
    kaldi_dir: typing.Union[str, Path],
    scopes: typing.Optional[typing.Set[str]] = None,
    eps: int = 0,
):
    # Convert to paths
    model_dir = Path(model_dir)
    graph_dir = Path(graph_dir)
    kaldi_dir = Path(kaldi_dir)

    intent_state_path = model_dir / "intent_states.txt"

    if scopes is None:
        scopes = set()

    # Loading mapping between intent edges and FST states
    edges_to_drop: typing.Set[typing.Tuple[str, str]] = set()
    with open(intent_state_path, "r", encoding="utf-8") as intent_state_file:
        for line in intent_state_file:
            line = line.strip()
            if not line:
                continue

            from_state, to_state, intent_name = line.split(maxsplit=2)

            if scopes:
                # [scope:intent]
                if ":" in intent_name:
                    intent_scope = intent_name.split(":", maxsplit=1)[0]
                else:
                    intent_scope = DEFAULT_SCOPE

                if intent_scope not in scopes:
                    edges_to_drop.add((from_state, to_state))

    # Extend PATH
    egs_utils_dir = kaldi_dir / "egs" / "wsj" / "s5" / "utils"
    extended_env = os.environ.copy()
    extended_env["PATH"] = (
        str(kaldi_dir) + ":" + str(egs_utils_dir) + ":" + extended_env["PATH"]
    )

    data_dir = model_dir / "data"
    lang_dir = data_dir / "lang"

    fst_path = lang_dir / "G.fst"
    fst_text_path = lang_dir / "G.fst.txt"
    original_fst_path = lang_dir / "G.fst.original"

    if not original_fst_path.is_file():
        shutil.copy(fst_path, original_fst_path)

    fst_path.unlink(missing_ok=True)

    try:
        # Prune FST states for out of scope intents
        fstprint_cmd = ["fstprint", str(original_fst_path)]

        # pylint: disable=consider-using-with
        fstprint_proc = subprocess.Popen(
            fstprint_cmd, stdout=subprocess.PIPE, universal_newlines=True
        )

        assert fstprint_proc.stdout is not None

        with open(fst_text_path, "w", encoding="utf-8") as fst_text_file:
            for line in fstprint_proc.stdout:
                line = line.strip()
                if not line:
                    continue

                line_parts = line.split()
                if len(line_parts) >= 4:
                    # Skip out of scope intent edges
                    from_state, to_state = line_parts[0], line_parts[0]
                    if (from_state, to_state) in edges_to_drop:
                        continue

                print(line, file=fst_text_file)

        # 2. fstcompile > G.fst
        compile_grammar = [
            "fstcompile",
            shlex.quote(str(fst_text_path)),
            shlex.quote(str(lang_dir / "G.fst.unsorted")),
        ]

        _LOGGER.debug(compile_grammar)
        subprocess.check_call(compile_grammar, cwd=model_dir, env=extended_env)

        arcsort = [
            "fstarcsort",
            "--sort_type=ilabel",
            shlex.quote(str(lang_dir / "G.fst.unsorted")),
            shlex.quote(str(lang_dir / "G.fst")),
        ]

        _LOGGER.debug(arcsort)
        subprocess.check_call(arcsort, cwd=model_dir, env=extended_env)
        os.unlink(lang_dir / "G.fst.unsorted")

        # 3. mkgraph.sh
        mkgraph = [
            "bash",
            str(egs_utils_dir / "mkgraph.sh"),
            "--self-loop-scale",
            "1.0",
            str(lang_dir),
            str(model_dir / "model"),
            str(graph_dir),
        ]
        _LOGGER.debug(mkgraph)
        subprocess.check_call(mkgraph, cwd=model_dir, env=extended_env)

        # 4. prepare_online_decoding.sh
        train_prepare_online_decoding(model_dir, lang_dir, kaldi_dir)
    finally:
        # Restore G.fst
        fst_path.unlink(missing_ok=True)
        shutil.copy(original_fst_path, fst_path)
