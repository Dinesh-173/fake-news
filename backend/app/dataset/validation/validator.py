"""Data validator evaluating records against schema, quality, and encoding rules."""

from collections import Counter

from backend.app.dataset.schemas import CanonicalNewsRecord, ValidationSummary


class DataValidator:
    """Validates canonical records and classifies them into VALID, WARNING, or INVALID."""

    MIN_TEXT_CHARS = 10
    MAX_TEXT_CHARS = 50000

    @classmethod
    def validate_record(cls, record: CanonicalNewsRecord) -> tuple[str, list[str]]:
        """
        Evaluates a single canonical record.
        Returns (status, notes) where status is 'VALID', 'WARNING', or 'INVALID'.
        """
        notes: list[str] = []
        is_invalid = False

        # 1. Mandatory Identity Fields
        if not record.record_id:
            notes.append("missing_record_id")
            is_invalid = True
        if not record.dataset_name:
            notes.append("missing_dataset_name")
            is_invalid = True
        if not record.original_id:
            notes.append("missing_original_id")
            is_invalid = True

        # 2. Text Quality & Boundaries
        text = record.text or ""
        text_len = len(text.strip())
        if text_len == 0:
            notes.append("empty_or_whitespace_text")
            is_invalid = True
        elif text_len < cls.MIN_TEXT_CHARS:
            notes.append(f"extremely_short_text_{text_len}_chars")
        elif text_len > cls.MAX_TEXT_CHARS:
            notes.append(f"exceeds_max_length_{text_len}_chars")

        # 3. Label Validity
        if not record.original_label or not str(record.original_label).strip():
            notes.append("missing_original_label")
            is_invalid = True

        # 4. Timestamp Parseability
        if record.published_at_raw and record.published_at_normalized is None:
            notes.append("unparseable_timestamp")

        # 5. Image Modality Consistency
        if record.image_path and not record.has_image:
            notes.append("image_path_specified_but_unreachable")

        if is_invalid:
            return "INVALID", notes
        elif len(notes) > 0:
            return "WARNING", notes
        return "VALID", []

    @classmethod
    def validate_dataset(
        cls, records: list[CanonicalNewsRecord]
    ) -> tuple[list[CanonicalNewsRecord], ValidationSummary]:
        """
        Validates an entire batch of records, attaching validation status and notes.
        Returns validated records and an aggregated ValidationSummary.
        """
        validated_records: list[CanonicalNewsRecord] = []
        warning_counter = Counter()
        invalid_counter = Counter()
        valid_count = 0
        warning_count = 0
        invalid_count = 0

        for r in records:
            status, notes = cls.validate_record(r)
            if status == "VALID":
                valid_count += 1
            elif status == "WARNING":
                warning_count += 1
                for note in notes:
                    warning_counter[note] += 1
            else:
                invalid_count += 1
                for note in notes:
                    invalid_counter[note] += 1

            validated_records.append(
                r.model_copy(
                    update={
                        "validation_status": status,
                        "validation_notes": notes,
                    }
                )
            )

        summary = ValidationSummary(
            total_records=len(records),
            valid_records=valid_count,
            warning_records=warning_count,
            invalid_records=invalid_count,
            warning_reasons=dict(warning_counter),
            invalid_reasons=dict(invalid_counter),
        )
        return validated_records, summary
