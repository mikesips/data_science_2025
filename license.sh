#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2025 Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: EUPL-1.2

year="2025"
author="Helmholtz Centre for Geosciences"
license_python="EUPL-1.2"

shopt -s globstar

for file in *. **/*.sh; 
do
reuse annotate --copyright "$author" --year "$year" --license "$license_python" $file
done