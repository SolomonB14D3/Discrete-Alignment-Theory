# 5-Minute Verification
To verify the 81.4% reduction and the $n=12$ peak locally:

1. **Build the Environment:**
   `docker build -t dat-verification .`

2. **Run the Analysis:**
   `docker run dat-verification python3 scripts/verify_manuscript_data.py`

3. **Check the Output:**
   If the terminal displays "✅ PASS" for both Pillar 1 and Pillar 3, the metrics in the manuscript are computationally substantiated.
