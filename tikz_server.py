from flask import Flask, request, jsonify, send_file
from io import BytesIO
from os import environ
from os.path import isfile, join
from re import MULTILINE, escape, search
from subprocess import CalledProcessError, DEVNULL, TimeoutExpired
from tempfile import NamedTemporaryFile, TemporaryDirectory
import base64

from pdfCropMargins import crop

# Reuse the check_output utility or define a simple wrapper
from subprocess import check_output as _check_output

def check_output(**kwargs):
    return _check_output(**kwargs)

app = Flask(__name__)

ENGINES = ["pdflatex", "lualatex", "xelatex"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/set_engines", methods=["POST"])
def set_engines():
    global ENGINES
    data = request.get_json()
    engines = data.get("engines", ENGINES)
    if isinstance(engines, str):
        engines = [engines]
    ENGINES = engines
    return jsonify({"engines": ENGINES})


@app.route("/compile", methods=["POST"])
def compile_tikz():
    data = request.get_json()
    code = data.get("code", "")
    timeout = data.get("timeout", 60)

    result = {
        "pdf": None,
        "cropped_pdf": None,
        "status": -1,
        "log": "",
        "error": None,
    }

    with TemporaryDirectory() as tmpdirname:
        with NamedTemporaryFile(dir=tmpdirname, buffering=0, suffix=".tex") as tmpfile:
            codelines = code.split("\n")
            codelines.insert(
                1,
                r"{cmd}\AtBeginDocument{{{cmd}}}".format(
                    cmd=r"\thispagestyle{empty}\pagestyle{empty}"
                ),
            )
            tmpfile.write("\n".join(codelines).encode())

            try:
                errorln = -1
                tmppdf = f"{tmpfile.name}.pdf"
                outpdf = join(tmpdirname, "tikz.pdf")
                open(f"{tmpfile.name}.bbl", "a").close()

                import pymupdf

                def try_save_last_page():
                    try:
                        doc = pymupdf.open(tmppdf)
                        doc.select([len(doc) - 1])
                        doc.save(outpdf)
                    except Exception:
                        pass

                for engine in ENGINES:
                    try:
                        _check_output(
                            cwd=tmpdirname,
                            timeout=timeout,
                            stderr=DEVNULL,
                            env=environ | dict(max_print_line="1000"),
                            args=[
                                "latexmk",
                                "-f",
                                "-nobibtex",
                                "-norc",
                                "-file-line-error",
                                "-interaction=nonstopmode",
                                f"-{engine}",
                                tmpfile.name,
                            ],
                        )
                    except (CalledProcessError, TimeoutExpired) as proc:
                        log = (getattr(proc, "output", b"") or b"").decode(
                            errors="ignore"
                        )
                        error = search(
                            rf"^{escape(tmpfile.name)}:(\d+):.+$", log, MULTILINE
                        )
                        linenr = int(error.group(1)) if error else 0
                        if linenr > errorln:
                            errorln = linenr
                            result.update(
                                status=getattr(proc, "returncode", -1), log=log
                            )
                            try_save_last_page()
                    else:
                        result.update(status=0, log="")
                        try_save_last_page()
                        break

                # Read the uncropped PDF if it exists
                if isfile(outpdf):
                    with open(outpdf, "rb") as f:
                        result["pdf"] = base64.b64encode(f.read()).decode("ascii")

                # Crop
                croppdf = f"{tmpfile.name}.crop"
                try:
                    crop(
                        ["-gsf", "-c", "gb", "-p", "0", "-a", "-1", "-o", croppdf, outpdf],
                        quiet=True,
                    )
                    if isfile(croppdf):
                        with open(croppdf, "rb") as f:
                            result["cropped_pdf"] = base64.b64encode(f.read()).decode(
                                "ascii"
                            )
                except Exception:
                    pass

            except FileNotFoundError:
                result["error"] = "Missing dependencies: Did you install TeX Live?"
            except RuntimeError:
                result["error"] = "PDF error during cropping"

    return jsonify(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TikZ Compilation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8070, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"Starting TikZ compilation server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)