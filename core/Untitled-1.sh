python3 compare_models.py --pred1 ../approaches/moral-values/output/Baseline-test.tsv --pred2 ../approaches/openness_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Baseline-test.tsv | tee ../approaches/openness_moral-values/output/comparison_MainBaseline_Baseline.txt

--

python3 compare_models.py --pred1 ../approaches/conservation_moral-values/output/1_Lex-LIWC-22_LingFeat_0.1_Baseline-test.tsv --pred2 ../approaches/conservation_moral-values/output/1_Lex-LIWC-22_LingFeat_0.1_Lex-Schwartz-test.tsv | tee ../approaches/conservation_moral-values/output/comparison_Baseline_Lex-Schwartz.txt