import faithfulness


def main():
    tests = [
        (
            "Under the Fourteenth Amendment, Meyer v. Nebraska recognized protected liberty.",
            {"meyer v. nebraska"},
        ),
        (
            "The Court cited Twining v. New Jersey, 211 U.S. 78.",
            {"twining v. new jersey"},
        ),
        (
            "United States v. James Daniel Good Real Property applied Mathews v. Eldridge.",
            {
                "united states v. james daniel good real property",
                "mathews v. eldridge",
            },
        ),
        (
            "# Liberty Interest Recognized in Meyer v. Nebraska\n\nUnder the Fourteenth Amendment's Due Process Clause...",
            {"meyer v. nebraska"},
        ),
        (
            "See Austin v. United States.",
            {"austin v. united states"},
        ),
        (
            "Cf. Gerstein v. Pugh.",
            {"gerstein v. pugh"},
        ),
        (
            "and Connecticut v. Doehr",
            {"connecticut v. doehr"},
        ),
        (
            "of Fuentes v. Shevin",
            {"fuentes v. shevin"},
        ),
    ]

    failures = 0

    for text, expected in tests:
        actual = faithfulness.cases_in(text)
        if actual != expected:
            failures += 1
            print("FAIL")
            print("TEXT:    ", text)
            print("EXPECTED:", expected)
            print("ACTUAL:  ", actual)
            print()

    if failures:
        raise SystemExit(f"{failures} parser test(s) failed.")

    print("✅ All faithfulness parser tests passed.")


if __name__ == "__main__":
    main()