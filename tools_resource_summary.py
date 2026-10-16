#!/usr/bin/env python3
"""Turn the raw CC-100 counts into the figures quoted in Section 3.6.

Inputs: results/round3/resource_counts.json (pa, complete) and
results/round4/resource_counts_round4.json (sr, hr, bs, ur, with the
fraction of each file read). Files read only in part are scaled up by that
fraction. Classifier error rates come from the FLORES validation in the
same files. Writes results/round4/resource_summary.json.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")


def main():
    r3 = json.load(open(os.path.join(R, "round3", "resource_counts.json")))
    r4 = json.load(open(os.path.join(R, "round4", "resource_counts_round4.json")))
    v = r4["classifier_validation"]
    out = {}

    pa = r3["cc100"]["pa"]
    assert pa["complete"], "the Punjabi file must be read in full"
    gur = pa["letters"]["gurmukhi"]
    ur = r4["cc100"]["ur"]
    assert ur["fraction_of_file"] == 1.0, "the Urdu file must be read in full"
    p = ur["arabic_letters_by_variety"]["punjabi"]
    u = ur["arabic_letters_by_variety"]["urdu"]
    fp, rec = v["urd_Arab"]["wrong"], v["pan_Shah"]["correct"]
    corrected = (p - fp * u) / rec
    rec_h = v["pumvr_human_shahmukhi"]["correct"]
    corrected_h = (p - fp * u) / rec_h
    out["punjabi"] = {
        "gurmukhi_letters": gur,
        "shahmukhi_letters_classified": p,
        "shahmukhi_letters_corrected": corrected,
        "shahmukhi_over_gurmukhi_raw": p / gur,
        "shahmukhi_over_gurmukhi_corrected": corrected / gur,
        "shahmukhi_letters_corrected_human_recall": corrected_h,
        "shahmukhi_over_gurmukhi_corrected_human_recall": corrected_h / gur,
        "urdu_letters": u,
        "pnb_file_in_cc100": r3["cc100"]["pnb"].get("status", "") .startswith("ok"),
        "correction": "subtract Urdu misread as Punjabi (FLORES false-positive "
                      "rate), divide by FLORES recall on Shahmukhi",
    }

    sr = r4["cc100"]["sr"]
    f = sr["fraction_of_file"]
    cyr = sr["letters"]["cyrillic"] / f
    lat_sr = sr["letters"]["latin"] / f
    h = v["hrv_Latn"]
    fpr = h["wrong"] / (h["wrong"] + h["correct"])     # Croatian read as ekavian

    def ekavian(d, frac):
        e = d["latin_letters_by_variety"]["ekavian"]
        j = d["latin_letters_by_variety"]["ijekavian"]
        tot = sum(d["latin_letters_by_variety"].values())
        true = max(0.0, e / (e + j) - fpr) / (1 - fpr)
        return true * tot / frac, e / (e + j)

    hr, bs = r4["cc100"]["hr"], r4["cc100"]["bs"]
    e_hr, raw_hr = ekavian(hr, hr["fraction_of_file"])
    e_bs, raw_bs = ekavian(bs, bs["fraction_of_file"])
    lat_all = hr["letters"]["latin"] / hr["fraction_of_file"] + bs["letters"]["latin"] + lat_sr
    serb_lat = lat_sr + e_hr + e_bs
    out["serbian"] = {
        "cyrillic_letters_est": cyr,
        "latin_letters_under_sr_est": lat_sr,
        "latin_share_of_sr_file": sr["letters"]["latin"] / (sr["letters"]["latin"] + sr["letters"]["cyrillic"]),
        "latin_serbo_croatian_letters_est": lat_all,
        "latin_serbo_croatian_over_serbian_cyrillic": lat_all / cyr,
        "ekavian_latin_letters_est": serb_lat,
        "ekavian_latin_over_serbian_cyrillic": serb_lat / cyr,
        "ekavian_share_of_decided_hr_raw": raw_hr,
        "croatian_false_ekavian_rate": fpr,
        "fractions_read": {"sr": f, "hr": hr["fraction_of_file"], "bs": bs["fraction_of_file"]},
    }
    out["classifier_validation"] = v
    path = os.path.join(R, "round4", "resource_summary.json")
    json.dump(out, open(path, "w"), indent=2)
    pj, sb = out["punjabi"], out["serbian"]
    print("Shahmukhi / Gurmukhi: %.2f%% raw, %.2f%% corrected"
          % (100 * pj["shahmukhi_over_gurmukhi_raw"], 100 * pj["shahmukhi_over_gurmukhi_corrected"]))
    print("Serbian Cyrillic %.2fB letters, Latin under sr %.2f%% of the file"
          % (sb["cyrillic_letters_est"] / 1e9, 100 * sb["latin_share_of_sr_file"]))
    print("Latin Serbo-Croatian text / Serbian Cyrillic: %.1f" % sb["latin_serbo_croatian_over_serbian_cyrillic"])
    print("Ekavian (Serbian-standard) Latin / Serbian Cyrillic: %.1f%%" % (100 * sb["ekavian_latin_over_serbian_cyrillic"]))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
