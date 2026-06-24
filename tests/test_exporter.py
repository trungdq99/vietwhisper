import io
import pytest
from utils.exporter import format_timestamp, export_to_markdown, export_to_docx, export_to_zip

def test_format_timestamp_valid():
    assert format_timestamp(0.0) == "[00:00:00.000]"
    assert format_timestamp(3661.123) == "[01:01:01.123]"
    assert format_timestamp(59.999) == "[00:00:59.999]"

def test_format_timestamp_negative():
    with pytest.raises(ValueError, match="Thời gian không được âm"):
        format_timestamp(-1.0)

def test_export_to_markdown_with_timestamps():
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Xin chào."},
        {"start": 3.0, "end": 5.5, "text": " Tôi là trợ lý AI."}
    ]
    bio = export_to_markdown(segments, include_timestamps=True)
    content = bio.getvalue().decode("utf-8")
    assert "**[00:00:00.000]** Xin chào." in content
    assert "**[00:00:03.000]** Tôi là trợ lý AI." in content

def test_export_to_markdown_without_timestamps():
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Xin chào."},
        {"start": 3.0, "end": 5.5, "text": " Tôi là trợ lý AI."}
    ]
    bio = export_to_markdown(segments, include_timestamps=False)
    content = bio.getvalue().decode("utf-8")
    assert "Xin chào." in content
    assert "Tôi là trợ lý AI." in content
    assert "[" not in content

def test_export_to_docx_with_timestamps():
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Xin chào."},
        {"start": 3.0, "end": 5.5, "text": " Tôi là trợ lý AI."}
    ]
    bio = export_to_docx(segments, include_timestamps=True)
    assert bio.getvalue().startswith(b"PK")  # DOCX files are zip archives under the hood

def test_export_to_docx_without_timestamps():
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Xin chào."},
        {"start": 3.0, "end": 5.5, "text": " Tôi là trợ lý AI."}
    ]
    bio = export_to_docx(segments, include_timestamps=False)
    assert bio.getvalue().startswith(b"PK")

def test_export_to_zip():
    files = {
        "file1.md": io.BytesIO(b"Hello MD"),
        "file2.docx": io.BytesIO(b"Hello DOCX")
    }
    zip_bio = export_to_zip(files)
    assert zip_bio.getvalue().startswith(b"PK")
