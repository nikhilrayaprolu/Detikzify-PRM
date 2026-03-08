import base64
from collections import namedtuple
from functools import cache, cached_property
from io import BytesIO
from os import environ
from re import MULTILINE, findall, search
from typing import Dict, List, Optional, Union

import requests
from PIL import Image
from pdf2image.pdf2image import convert_from_bytes
import pymupdf
from transformers.utils import logging

from ..util import expand, redact as redact_text

logger = logging.get_logger("transformers")

_DEFAULT_SERVER_URL = environ.get("TIKZ_SERVER_URL", "http://10.2.135.88:8070/")


class TikzDocument:
    engines: List[str] = ["pdflatex", "lualatex", "xelatex"]
    server_url: str = _DEFAULT_SERVER_URL
    Output = namedtuple("Output", ["pdf", "status", "log"], defaults=[None, -1, ""])

    def __init__(self, code: str, timeout: Optional[int] = 60):
        self.code = code
        self.timeout = timeout
        self.compile = cache(self.compile)

    @classmethod
    def set_server_url(cls, url: str):
        cls.server_url = url.rstrip("/")

    @classmethod
    def set_engines(cls, engines: Union[str, list]):
        cls.engines = [engines] if isinstance(engines, str) else engines
        try:
            requests.post(
                f"{cls.server_url}/set_engines",
                json={"engines": cls.engines},
                timeout=10,
            )
        except requests.RequestException:
            logger.warning("Could not sync engines to server")

    @property
    def status(self) -> int:
        return self.compile().status

    @property
    def pdf(self) -> Optional[pymupdf.Document]:
        return self.compile().pdf

    @property
    def log(self) -> str:
        return self.compile().log

    @property
    def compiled_with_errors(self) -> bool:
        return self.status != 0

    @property
    def errors(self, rootfile: Optional[str] = None) -> Dict[int, str]:
        if self.compiled_with_errors:
            if not rootfile and (
                match := search(r"^\((.+)$", self.log, MULTILINE)
            ):
                rootfile = match.group(1)
            else:
                ValueError("rootfile not found!")

            errors = dict()
            for file, line, error in findall(
                r"^(.+):(\d+):(.+)$", self.log, MULTILINE
            ):
                if file == rootfile:
                    errors[int(line)] = error.strip()
                else:
                    errors[0] = error.strip()

            return errors or {
                0: "Fatal error occurred, no output PDF file produced!"
            }
        return dict()

    @cached_property
    def is_rasterizable(self) -> bool:
        return self.rasterize() is not None

    @cached_property
    def has_content(self) -> bool:
        return (img := self.rasterize()) is not None and img.getcolors(1) is None

    def compile(self) -> "Output":
        output: dict = {}

        try:
            resp_timeout = (
                max(self.timeout + 30, 120) if self.timeout else 300
            )
            response = requests.post(
                f"{self.server_url}/compile",
                json={
                    "code": self.code,
                    "timeout": self.timeout,
                },
                timeout=resp_timeout,
            )
            response.raise_for_status()
            data = response.json()

            output["status"] = data.get("status", -1)
            output["log"] = data.get("log", "")

            if err := data.get("error"):
                logger.error(f"Server error: {err}")

            # Check what PDFs the server returned
            has_cropped = bool(data.get("cropped_pdf"))
            has_uncropped = bool(data.get("pdf"))

            logger.info(
                f"Compile response: status={output['status']}, "
                f"has_pdf={has_uncropped}, has_cropped_pdf={has_cropped}"
            )

            # Prefer cropped, fall back to uncropped
            pdf_b64 = data.get("cropped_pdf") or data.get("pdf")

            if pdf_b64:
                try:
                    pdf_bytes = base64.b64decode(pdf_b64)
                    if len(pdf_bytes) == 0:
                        logger.warning("Decoded PDF is empty (0 bytes)")
                    else:
                        output["pdf"] = pymupdf.open(
                            stream=pdf_bytes, filetype="pdf"
                        )
                        logger.info(
                            f"Loaded PDF: {len(pdf_bytes)} bytes, "
                            f"{len(output['pdf'])} pages"
                        )
                except Exception as exc:
                    logger.error(f"Failed to decode/open PDF: {exc}")

        except requests.ConnectionError:
            logger.error(
                f"Cannot connect to TikZ compilation server at "
                f"{self.server_url}. Is the server running?"
            )
        except requests.Timeout:
            logger.error("Request to compilation server timed out.")
        except requests.RequestException as exc:
            logger.error(f"Compilation server request failed: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error during remote compilation: {exc}")

        if output.get("status") == 0 and not output.get("pdf"):
            logger.warning(
                "Compilation succeeded (status=0) but no PDF was returned. "
                "Check server logs for details."
            )

        return self.Output(**output)

    def rasterize(
        self,
        size=420,
        expand_to_square=True,
        redact=False,
        **redact_kwargs,
    ) -> Optional[Image.Image]:
        if pdf := self.pdf:
            try:
                if redact:
                    pdf = redact_text(pdf, **redact_kwargs)
                image = convert_from_bytes(
                    pdf.tobytes(), size=size, single_file=True
                )[0]
                if expand_to_square:
                    return expand(image, size)
                return image
            except Exception as exc:
                logger.error(f"Rasterization failed: {exc}")
                return None
        return None

    def save(self, filename: str, *args, **kwargs):
        ext = filename.rsplit(".", 1)[-1].lower()

        if ext == "tex":
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.code)
            return

        if ext == "pdf":
            if self.pdf is not None:
                with open(filename, "wb") as f:
                    f.write(self.pdf.tobytes())
                return
            raise ValueError(
                "No PDF available — compilation may have failed "
                "or the server returned no PDF."
            )

        # Image formats (png, jpg, etc.)
        img = self.rasterize(*args, **kwargs)
        if img is not None:
            img.save(buf := BytesIO(), format=ext)
            with open(filename, "wb") as f:
                f.write(buf.getvalue())
            return

        if self.pdf is None:
            raise ValueError(
                f"Cannot save as '{ext}': no PDF available for rasterization."
            )
        raise ValueError(
            f"Cannot save as '{ext}': rasterization failed "
            f"(PDF exists but could not be converted to image)."
        )