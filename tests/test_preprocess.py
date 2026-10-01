from src.preprocess import mask_entities, mask_sensitive_data


def test_masks_email_phone_and_long_sensitive_number():
    masked = mask_sensitive_data("Email a@shop.test or call +91 98765 43210; payment 4111 1111 1111 1111")
    assert "a@shop.test" not in masked
    assert "98765" not in masked
    assert "4111 1111" not in masked
    assert "[EMAIL]" in masked and "[SENSITIVE_NUMBER]" in masked


def test_masks_stored_operational_references_but_keeps_safe_entities():
    entities = mask_entities({"transaction_ids": ["TXN-123456"], "tracking_ids": ["TRACK-123456"], "amounts": ["$20"]})
    assert entities["transaction_ids"] == ["***3456"]
    assert entities["tracking_ids"] == ["***3456"]
