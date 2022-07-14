#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Submission form mechanism.
"""

import mechanize
import sys
import os

assert len(sys.argv) == 2, f"Usage: {sys.argv[0]} <zip_file>"
team_name = "Beach Nerds"
team_token = "PraiaDaTorreira"
submission_zip = sys.argv[1]
submission_form_url = "http://34.138.95.36:8000"

assert os.path.isfile(submission_zip)
assert team_token != ""
assert team_name != ""

validate = None
while validate not in ["y", "n"]:
    validate = input("[*] Validate only? [Y/n]")
    validate = validate.lower()
br = mechanize.Browser()
br.set_handle_robots(False)
br.open(submission_form_url)
br.select_form(nr=0)
br["team_name"] = team_name
br["team_token"] = team_token
if validate:
    br.find_control("validate_submission").items[0].selected=True
br.add_file(open(submission_zip, "rb"), content_type="application/octet-stream", filename="submission.zip")
res = br.submit()
content = res.read()
print(content)

