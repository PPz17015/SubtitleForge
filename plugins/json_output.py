from pathlib import Path

from api.plugin_system import OutputPlugin


class JSONOutputPlugin(OutputPlugin):
    @property
    def name(self) -> str:
        return "json_output"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Export subtitles to JSON format"

    def write(self, segments: list, output_path: Path, **kwargs) -> bool:
        import json

        data = []
        for seg in segments:
            data.append({
                "index": seg.index,
                "start": seg.start_time,
                "end": seg.end_time,
                "text": seg.text,
                "translation": getattr(seg, "translation", ""),
                "speaker": getattr(seg, "speaker", "")
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True


def register():
    return JSONOutputPlugin()
