#!/bin/bash
# Run RFC compliance tests for PyShorthand

set -e

echo "======================================================================"
echo "PyShorthand RFC v1.3.1 Compliance Test Suite"
echo "======================================================================"
echo ""

# Run tests with verbose output
python3 -m unittest tests.compliance.test_rfc_compliance -v

echo ""
echo "======================================================================"
echo "Compliance testing complete!"
echo "======================================================================"
