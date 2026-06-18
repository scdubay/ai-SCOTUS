"""
expanded_eval_questions.py

Larger evaluation set for the Supreme Court RAG pipeline.

Goals:
1. Test routing.
2. Test case-scoped retrieval.
3. Test majority / concurrence / dissent distinction.
4. Test precedent extraction.
5. Test negative / trap questions.
6. Test whether the model stays inside the provided case material.
"""

EXPANDED_EVAL_QUESTIONS = [
    # ---------------------------------------------------------------------
    # Meyer v. Nebraska
    # ---------------------------------------------------------------------
    {
        "case_title": "Meyer v. Nebraska",
        "question": "What liberty interest did Meyer v. Nebraska recognize under the Fourteenth Amendment?",
        "expected_terms": [
            "Fourteenth Amendment",
            "liberty",
            "acquire useful knowledge",
            "parents",
            "education",
        ],
    },
    {
        "case_title": "Meyer v. Nebraska",
        "question": "What right of parents did Meyer v. Nebraska recognize?",
        "expected_terms": [
            "parents",
            "control",
            "education",
            "children",
        ],
    },
    {
        "case_title": "Meyer v. Nebraska",
        "question": "Why did the Court strike down the Nebraska statute?",
        "expected_terms": [
            "arbitrary",
            "reasonable relation",
            "liberty",
            "German",
            "police power",
        ],
    },
    {
        "case_title": "Meyer v. Nebraska",
        "question": "How did the Court evaluate Nebraska's exercise of police power?",
        "expected_terms": [
            "police power",
            "not final",
            "courts",
            "arbitrary",
            "reasonable relation",
        ],
    },
    {
        "case_title": "Meyer v. Nebraska",
        "question": "What precedents did Meyer v. Nebraska rely upon?",
        "expected_terms": [
            "Twining",
            "Allgeyer",
            "Lochner",
            "Adams",
            "Lawton",
        ],
    },
    {
        "case_title": "Meyer v. Nebraska",
        "question": "Did the Court question Nebraska's power to require instruction in English?",
        "expected_terms": [
            "no",
            "instruction in English",
            "school attendance",
            "curriculum",
        ],
    },
    {
        "case_title": "Meyer v. Nebraska",
        "question": "Did Meyer v. Nebraska apply strict scrutiny?",
        "expected_terms": [
            "no",
            "strict scrutiny",
            "arbitrary",
            "reasonable relation",
            "police power",
        ],
    },
    {
        "case_title": "Meyer v. Nebraska",
        "question": "How did Meyer distinguish permissible education regulation from the statute at issue?",
        "expected_terms": [
            "school attendance",
            "instruction in English",
            "curriculum",
            "foreign language",
            "parents",
        ],
    },

    # ---------------------------------------------------------------------
    # United States v. James Daniel Good Real Property
    # ---------------------------------------------------------------------
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "What balancing test did the Court apply in United States v. James Daniel Good Real Property?",
        "expected_terms": [
            "Mathews",
            "private interest",
            "risk of erroneous deprivation",
            "Government's interest",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "Why did the Court require notice and a hearing before seizure of real property?",
        "expected_terms": [
            "notice",
            "hearing",
            "real property",
            "due process",
            "exigent circumstances",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "When may the Government seize real property without prior notice and hearing?",
        "expected_terms": [
            "exigent circumstances",
            "lis pendens",
            "restraining order",
            "bond",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "Why did the Court distinguish real property from personal property?",
        "expected_terms": [
            "real property",
            "cannot abscond",
            "cannot be moved",
            "Calero-Toledo",
            "jurisdiction",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "What government interests did the Court consider insufficient to justify seizure without notice?",
        "expected_terms": [
            "jurisdiction",
            "sale",
            "destruction",
            "illegal use",
            "revenue",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "What precedents did the Court rely upon in United States v. James Daniel Good Real Property?",
        "expected_terms": [
            "Mathews",
            "Fuentes",
            "Calero-Toledo",
            "Mullane",
            "Doehr",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "What was Justice O'Connor's primary criticism of the majority opinion?",
        "expected_terms": [
            "O'Connor",
            "Calero-Toledo",
            "real property",
            "personal property",
            "precedent",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "What did Justice Thomas agree with and disagree with in the Court's due process ruling?",
        "expected_terms": [
            "Thomas",
            "property rights",
            "civil forfeiture",
            "due process",
            "dissent",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "What was Chief Justice Rehnquist's objection to applying Mathews v. Eldridge?",
        "expected_terms": [
            "Rehnquist",
            "Mathews",
            "Fourth Amendment",
            "civil forfeiture",
            "historical practice",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "Did the Court hold that the forfeiture action was untimely?",
        "expected_terms": [
            "no",
            "statute of limitations",
            "timely",
            "internal timing",
            "customs",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "Did the Court hold that all civil forfeiture is unconstitutional?",
        "expected_terms": [
            "no",
            "real property",
            "notice",
            "hearing",
            "exigent circumstances",
        ],
    },

    # ---------------------------------------------------------------------
    # Miranda v. Arizona
    # ---------------------------------------------------------------------
    {
        "case_title": "Miranda v. Arizona",
        "question": "What warnings did Miranda v. Arizona require before custodial interrogation?",
        "expected_terms": [
            "right to remain silent",
            "counsel",
            "custodial interrogation",
            "waiver",
        ],
    },
    {
        "case_title": "Miranda v. Arizona",
        "question": "Why did the Court conclude that custodial interrogation requires procedural safeguards?",
        "expected_terms": [
            "custodial",
            "interrogation",
            "self-incrimination",
            "compulsion",
            "safeguards",
        ],
    },
    {
        "case_title": "Miranda v. Arizona",
        "question": "What did the Court say about waiver of the rights described in Miranda?",
        "expected_terms": [
            "waiver",
            "voluntary",
            "knowing",
            "intelligent",
            "rights",
        ],
    },
    {
        "case_title": "Miranda v. Arizona",
        "question": "How did the dissent criticize the majority's rule?",
        "expected_terms": [
            "dissent",
            "police",
            "confessions",
            "Constitution",
            "law enforcement",
        ],
    },
    {
        "case_title": "Miranda v. Arizona",
        "question": "Which Justices dissented in Miranda v. Arizona?",
        "expected_terms": [
            "Harlan",
            "White",
            "Clark",
            "dissent",
        ],
    },

    # ---------------------------------------------------------------------
    # Brown v. Board of Education
    # ---------------------------------------------------------------------
    {
        "case_title": "Brown v. Board of Education",
        "question": "Why did the Court reject separate but equal in public education?",
        "expected_terms": [
            "separate",
            "equal",
            "public education",
            "inferiority",
            "Fourteenth Amendment",
        ],
    },
    {
        "case_title": "Brown v. Board of Education",
        "question": "What role did education play in the Court's reasoning in Brown?",
        "expected_terms": [
            "education",
            "citizenship",
            "opportunity",
            "public schools",
            "importance",
        ],
    },
    {
        "case_title": "Brown v. Board of Education",
        "question": "Did Brown rely only on tangible equality of school facilities?",
        "expected_terms": [
            "no",
            "intangible",
            "segregation",
            "inferiority",
            "education",
        ],
    },
    {
        "case_title": "Brown v. Board of Education",
        "question": "What constitutional provision did the Court apply in Brown?",
        "expected_terms": [
            "Fourteenth Amendment",
            "Equal Protection",
            "public education",
        ],
    },

    # ---------------------------------------------------------------------
    # Gideon v. Wainwright
    # ---------------------------------------------------------------------
    {
        "case_title": "Gideon v. Wainwright",
        "question": "What right did Gideon v. Wainwright recognize?",
        "expected_terms": [
            "counsel",
            "Sixth Amendment",
            "Fourteenth Amendment",
            "state",
            "criminal",
        ],
    },
    {
        "case_title": "Gideon v. Wainwright",
        "question": "Why did the Court overrule Betts v. Brady?",
        "expected_terms": [
            "Betts",
            "Brady",
            "counsel",
            "fundamental",
            "fair trial",
        ],
    },
    {
        "case_title": "Gideon v. Wainwright",
        "question": "How did the Court describe the importance of counsel in a criminal prosecution?",
        "expected_terms": [
            "lawyer",
            "fair trial",
            "criminal prosecution",
            "fundamental",
        ],
    },
    {
        "case_title": "Gideon v. Wainwright",
        "question": "Did Gideon apply to state criminal prosecutions?",
        "expected_terms": [
            "state",
            "Fourteenth Amendment",
            "Sixth Amendment",
            "counsel",
        ],
    },

    # ---------------------------------------------------------------------
    # Marbury v. Madison
    # ---------------------------------------------------------------------
    {
        "case_title": "Marbury v. Madison",
        "question": "What did the Court say about a legal right without a remedy?",
        "expected_terms": [
            "right",
            "remedy",
            "laws",
            "government",
        ],
    },
    {
        "case_title": "Marbury v. Madison",
        "question": "What distinction did the Court draw between political acts and legally reviewable duties?",
        "expected_terms": [
            "political",
            "duty",
            "rights",
            "remedy",
            "discretion",
        ],
    },
    {
        "case_title": "Marbury v. Madison",
        "question": "Why did the Court hold that it could not issue the writ of mandamus?",
        "expected_terms": [
            "mandamus",
            "jurisdiction",
            "Constitution",
            "Judiciary Act",
        ],
    },
    {
        "case_title": "Marbury v. Madison",
        "question": "What principle of judicial review did Marbury establish?",
        "expected_terms": [
            "judicial review",
            "Constitution",
            "law",
            "courts",
            "void",
        ],
    },

    # ---------------------------------------------------------------------
    # Cross-case / trap questions
    # ---------------------------------------------------------------------
    {
        "case_title": "Meyer v. Nebraska",
        "question": "How did Meyer v. Nebraska apply the Mathews v. Eldridge balancing test?",
        "expected_terms": [
            "did not",
            "Mathews",
            "not in Meyer",
            "reasonable relation",
        ],
    },
    {
        "case_title": "United States v. James Daniel Good Real Property",
        "question": "Did Good Real Property overrule Miranda v. Arizona?",
        "expected_terms": [
            "no",
            "Miranda",
            "not addressed",
            "civil forfeiture",
            "due process",
        ],
    },
    {
        "case_title": "Brown v. Board of Education",
        "question": "Did Brown decide the right to appointed counsel in criminal cases?",
        "expected_terms": [
            "no",
            "counsel",
            "criminal",
            "education",
            "equal protection",
        ],
    },
]