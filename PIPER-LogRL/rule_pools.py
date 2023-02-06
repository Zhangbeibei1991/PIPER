"""
Conjunctive table
    alpha ^ beta -> beta
"""

tbd_all = ["VAGUE", "BEFORE", "AFTER", "SIMULTANEOUS", "INCLUDES", "IS_INCLUDED"]
tbd = {
    ("BEFORE", "BEFORE"): {"yes": ["BEFORE"], "not": ["VAGUE", "AFTER", "SIMULTANEOUS", "INCLUDES", "IS_INCLUDED"]},
    ("AFTER", "AFTER"): {"yes": ["AFTER"], "not": ["VAGUE", "BEFORE", "SIMULTANEOUS", "INCLUDES", "IS_INCLUDED"]},
    ("SIMULTANEOUS", "SIMULTANEOUS"): {"yes": ["SIMULTANEOUS"],
                                       "not": ["VAGUE", "BEFORE", "AFTER", "INCLUDES", "IS_INCLUDED"]},
    ("INCLUDES", "INCLUDES"): {"yes": ["INCLUDES"], "not": ["VAGUE", "BEFORE", "AFTER", "SIMULTANEOUS", "IS_INCLUDED"]},
    ("IS_INCLUDED", "IS_INCLUDED"): {"yes": ["IS_INCLUDED"],
                                     "not": ["VAGUE", "BEFORE", "AFTER", "SIMULTANEOUS", "INCLUDES"]},
    ("VAGUE", "VAGUE"): {"yes": ["VAGUE"], "not": ["BEFORE", "AFTER", "SIMULTANEOUS", "INCLUDES", "IS_INCLUDED"]},
    ("BEFORE", "VAGUE"): {"yes": ["BEFORE", "INCLUDES", "IS_INCLUDED", "VAGUE"], "not": ["AFTER", "SIMULTANEOUS"]},
    ("BEFORE", "INCLUDES"): {"yes": ["BEFORE", "INCLUDES", "VAGUE"], "not": ["AFTER", "SIMULTANEOUS", "IS_INCLUDED"]},
    ("BEFORE", "IS_INCLUDED"): {"yes": ["BEFORE", "IS_INCLUDED", "VAGUE"],
                                "not": ["AFTER", "SIMULTANEOUS", "INCLUDES"]},
    ("AFTER", "VAGUE"): {"yes": ["AFTER", "INCLUDES", "IS_INCLUDED", "VAGUE"], "not": ["BEFORE", "SIMULTANEOUS"]},
    ("AFTER", "INCLUDES"): {"yes": ["AFTER", "INCLUDES", "VAGUE"], "not": ["BEFORE", "SIMULTANEOUS", "IS_INCLUDED"]},
    ("AFTER", "IS_INCLUDED"): {"yes": ["AFTER", "IS_INCLUDED", "VAGUE"], "not": ["BEFORE", "SIMULTANEOUS", "INCLUDES"]},
    ("INCLUDES", "VAGUE"): {"yes": ["VAGUE", "BEFORE", "AFTER", "INCLUDES"], "not": ["IS_INCLUDED", "SIMULTANEOUS"]},
    ("INCLUDES", "BEFORE"): {"yes": ["BEFORE", "INCLUDES", "VAGUE"], "not": ["AFTER", "SIMULTANEOUS", "IS_INCLUDED"]},
    ("INCLUDES", "AFTER"): {"yes": ["AFTER", "INCLUDES", "VAGUE"], "not": ["BEFORE", "SIMULTANEOUS", "IS_INCLUDED"]},
    ("IS_INCLUDED", "VAGUE"): {"yes": ["VAGUE", "BEFORE", "AFTER", "IS_INCLUDED"], "not": ["INCLUDES", "SIMULTANEOUS"]},
    ("IS_INCLUDED", "BEFORE"): {"yes": ["BEFORE", "IS_INCLUDED", "VAGUE"],
                                "not": ["AFTER", "SIMULTANEOUS", "INCLUDES"]},
    ("IS_INCLUDED", "AFTER"): {"yes": ["AFTER", "IS_INCLUDED", "VAGUE"], "not": ["BEFORE", "SIMULTANEOUS", "INCLUDES"]},
    ("BEFORE", "SIMULTANEOUS"): {"yes": ["BEFORE"],
                                 "not": ["VAGUE", "SIMULTANEOUS", "AFTER", "INCLUDES", "IS_INCLUDED"]},
    ("AFTER", "SIMULTANEOUS"): {"yes": ["AFTER"],
                                "not": ["VAGUE", "SIMULTANEOUS", "BEFORE", "INCLUDES", "IS_INCLUDED"]},
    ("INCLUDES", "SIMULTANEOUS"): {"yes": ["INCLUDES"],
                                   "not": ["VAGUE", "SIMULTANEOUS", "BEFORE", "AFTER", "IS_INCLUDED"]},
    ("IS_INCLUDED", "SIMULTANEOUS"): {"yes": ["IS_INCLUDED"],
                                      "not": ["VAGUE", "SIMULTANEOUS", "BEFORE", "AFTER", "INCLUDES"]},
    # REVERSED VAGUE
    ("VAGUE", "BEFORE"): {"yes": ["BEFORE", "INCLUDES", "IS_INCLUDED", "VAGUE"], "not": ["AFTER", "SIMULTANEOUS"]},
    ("VAGUE", "AFTER"): {"yes": ["AFTER", "INCLUDES", "IS_INCLUDED", "VAGUE"], "not": ["BEFORE", "SIMULTANEOUS"]},
    ("VAGUE", "INCLUDES"): {"yes": ["VAGUE", "BEFORE", "AFTER", "INCLUDES"], "not": ["IS_INCLUDED", "SIMULTANEOUS"]},
    ("VAGUE", "IS_INCLUDED"): {"yes": ["VAGUE", "BEFORE", "AFTER", "IS_INCLUDED"], "not": ["INCLUDES", "SIMULTANEOUS"]},
}

matres_all = ["VAGUE", "BEFORE", "AFTER", "SIMULTANEOUS"]
matres = {
    ("BEFORE", "BEFORE"): {"yes": ["BEFORE"], "not": ["VAGUE", "AFTER", "SIMULTANEOUS"]},
    ("AFTER", "AFTER"): {"yes": ["AFTER"], "not": ["VAGUE", "BEFORE", "SIMULTANEOUS"]},
    ("SIMULTANEOUS", "SIMULTANEOUS"): {"yes": ["SIMULTANEOUS"], "not": ["VAGUE", "BEFORE", "AFTER"]},
    ("BEFORE", "VAGUE"): {"yes": ["BEFORE", "VAGUE"], "not": ["AFTER", "SIMULTANEOUS"]},
    ("AFTER", "VAGUE"): {"yes": ["AFTER", "VAGUE"], "not": ["BEFORE", "SIMULTANEOUS"]},
    ("BEFORE", "SIMULTANEOUS"): {"yes": ["BEFORE"], "not": ["VAGUE", "AFTER", "SIMULTANEOUS"]},
    ("AFTER", "SIMULTANEOUS"): {"yes": ["AFTER"], "not": ["VAGUE", "BEFORE", "SIMULTANEOUS"]},
    # REVERSED VAGUE
    ("VAGUE", "BEFORE"): {"yes": ["BEFORE", "VAGUE"], "not": ["AFTER", "SIMULTANEOUS"]},
    ("VAGUE", "AFTER"): {"yes": ["AFTER", "VAGUE"], "not": ["BEFORE", "SIMULTANEOUS"]},

}
