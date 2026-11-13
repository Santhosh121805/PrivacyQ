from src.m1_data.codebook import parse_codebook, plain_label

SAMPLE = '''label var sc_racer  "Race of Selected Child, Recode"
label var fpl_i1  "Family Poverty Ratio, First Implicate"
label define sc_racer_lab  1  "White alone"
    label define sc_racer_lab  2  "Black or African American alone", add
    label define sc_racer_lab  .m "No valid response", add
cap label values sc_racer sc_racer_lab
'''


def test_codebook_labels_raw_and_one_hot_columns(tmp_path):
    f = tmp_path / "x.do"
    f.write_text(SAMPLE, encoding="latin-1")
    var, val = parse_codebook(f)
    assert plain_label("fpl_i1", var, val) == "Family Poverty Ratio, First Implicate"
    assert plain_label("sc_racer_c2", var, val) == "Race of Selected Child, Recode: Black or African American alone"
    assert plain_label("sc_racer_c9", var, val) == "Race of Selected Child, Recode: code 9"
    assert plain_label("unknown_col", var, val) == "unknown_col"            # never invented


def test_display_value_decodes_loaded_values_to_codebook_labels():
    from src.m1_data.codebook import display_value
    vals = {"sc_sex": {1: "Male", 2: "Female"}, "sc_racer": {1: "White alone", 3: "Other"},
            "sc_hispanic_r": {1: "Hispanic or Latino Origin", 2: "Not Hispanic or Latino Origin"},
            "k2q01": {1: "Excellent", 5: "Poor"}, "fpl_i1": {50: "50 or less", 400: "400 or more"}, "ace10": {1: "Yes", 2: "No"}}
    assert display_value("sc_sex", 1.0, "binary", vals)[0] == "Male"
    assert display_value("sc_sex", 0.0, "binary", vals)[0] == "Female"
    assert display_value("sc_racer", "c3", "nominal", vals)[0] == "Other"
    assert display_value("sc_hispanic_r", 1.0, "binary", vals)[:2] == ("Yes", "raw value 1 (NSCH code 1): Hispanic or Latino Origin")
    assert display_value("ace10", 0.0, "binary", vals)[0] == "No"
    assert display_value("k2q01", 1.0, "ordinal", vals)[0] == "Excellent"
    assert display_value("fpl_i1", 400.0, "ordinal", vals)[0] == "400% or more of federal poverty level"
    assert display_value("fpl_i1", 297.0, "ordinal", vals)[0] == "297% of federal poverty level"
    assert display_value("sc_age_years", 12.0, "ordinal", vals)[0] == "12"
    assert display_value("k2q01", float("nan"), "ordinal", vals)[0] == "—"
    text, tip, known = display_value("k2q01", 9.0, "ordinal", vals)                     # unknown code: shown, flagged
    assert (text, known) == ("9", False) and "not in the NSCH codebook" in tip
