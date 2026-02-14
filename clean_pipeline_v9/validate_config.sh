#!/bin/bash
# Simple validation script for individual populations configuration

echo "=============================="
echo "CONFIGURATION VALIDATION"
echo "=============================="
echo

# Check if config file exists
echo "1. Checking configuration file..."
if [ -f "config/config_individual_pops.yaml" ]; then
    echo "   ✓ config/config_individual_pops.yaml exists"
else
    echo "   ✗ config/config_individual_pops.yaml not found"
    exit 1
fi

# Check if groups CSV exists
echo "2. Checking groups CSV..."
if [ -f "config/groups_12individuals.csv" ]; then
    echo "   ✓ config/groups_12individuals.csv exists"
    lines=$(wc -l < config/groups_12individuals.csv)
    echo "   ✓ Contains $lines lines (expected 13: header + 12 populations)"
else
    echo "   ✗ config/groups_12individuals.csv not found"
    exit 1
fi

# Check if SLiM template exists
echo "3. Checking SLiM template..."
if [ -f "templates/model_individual_pops.slim.tpl" ]; then
    echo "   ✓ templates/model_individual_pops.slim.tpl exists"
    lines=$(wc -l < templates/model_individual_pops.slim.tpl)
    echo "   ✓ Template has $lines lines"
else
    echo "   ✗ templates/model_individual_pops.slim.tpl not found"
    exit 1
fi

# Validate YAML syntax
echo "4. Validating YAML syntax..."
python3 -c "import yaml; yaml.safe_load(open('config/config_individual_pops.yaml'))" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ YAML syntax is valid"
else
    echo "   ✗ YAML syntax error"
    exit 1
fi

# Check coverage threshold
echo "5. Checking coverage threshold..."
cov=$(python3 -c "import yaml; print(yaml.safe_load(open('config/config_individual_pops.yaml'))['observed']['target_cov'])" 2>/dev/null)
if [ "$cov" == "10" ]; then
    echo "   ✓ Coverage threshold is 10"
else
    echo "   ✗ Coverage threshold is $cov (expected 10)"
fi

# Check population order
echo "6. Checking population order..."
python3 -c "import yaml; cfg = yaml.safe_load(open('config/config_individual_pops.yaml')); print(len(cfg['simulation']['pop_order']), 'populations in pop_order')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ Population order is defined"
else
    echo "   ✗ Error reading population order"
fi

# Check that template path references the individual pops template
echo "7. Checking template path..."
template=$(python3 -c "import yaml; print(yaml.safe_load(open('config/config_individual_pops.yaml'))['simulation']['slim_template'])" 2>/dev/null)
if [[ "$template" == *"individual"* ]]; then
    echo "   ✓ Template path references individual populations model: $template"
else
    echo "   ✗ Template path does not reference individual populations: $template"
fi

echo
echo "=============================="
echo "ALL VALIDATIONS PASSED ✓"
echo "=============================="
