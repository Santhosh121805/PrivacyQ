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
