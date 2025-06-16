# echo "=========="
# echo "=========="
# echo "===== Direct ====="
# echo "=========="
# echo "=========="
# echo ""
# echo "----- Comparing fixed baseline with fixed best model 1 -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/moral-values/output/Baseline-test.tsv --pred2 ../approaches/moral-values/output/Previous-Sentences-2-test.tsv --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/self-enh_moral-values/output/comparison-direct-fixed_baseline-fixed_model-1.txt
# echo ""
# echo "----- Comparing fixed baseline with fixed best model 2 -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/moral-values/output/Baseline-test.tsv --pred2 ../approaches/moral-values/output/Lex-LIWC-22-test.tsv --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/self-enh_moral-values/output/comparison-direct-fixed_baseline-fixed_model-2.txt
# echo ""
# echo "----- Comparing fixed baseline with tuned baseline -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/moral-values/output/Baseline-test.tsv --pred2 ../approaches/moral-values/output/Baseline-test.tsv --th2 0.3 --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/self-enh_moral-values/output/comparison-direct-fixed_baseline-tuned_baseline.txt
# echo ""
# echo "----- Comparing tuned baseline with tuned model -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/moral-values/output/Baseline-test.tsv --th1 0.3 --pred2 ../approaches/moral-values/output/Previous-Sentences-2-test.tsv --th2 0.15 --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/self-enh_moral-values/output/comparison-direct-tuned_baseline-tuned_model.txt
# 
# echo "=========="
# echo "=========="
# echo "===== Presence ====="
# echo "=========="
# echo "=========="
# echo ""
# echo "----- Comparing fixed baseline with fixed best model -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/p_moral-values/output/1_Previous-Sentences-2-Lex-LIWC-22_0.1_Baseline-test.tsv --pred2 ../approaches/p_moral-values/output/1_Previous-Sentences-2-Lex-LIWC-22_0.1_Lex-MJD-test.tsv --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/self-enh_moral-values/output/comparison-direct-fixed_baseline-fixed_model.txt
# echo ""
# echo "----- Comparing fixed baseline with tuned baseline -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/p_moral-values/output/1_Previous-Sentences-2-Lex-LIWC-22_0.1_Baseline-test.tsv --pred2 ../approaches/p_moral-values/output/1_Previous-Sentences-2-Lex-LIWC-22_0.1_Baseline-test.tsv --th2 0.3 --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/self-enh_moral-values/output/comparison-presence-fixed_baseline-tuned_baseline.txt
# echo ""
# echo "----- Comparing tuned baseline with tuned model -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/p_moral-values/output/1_Previous-Sentences-2-Lex-LIWC-22_0.1_Baseline-test.tsv --th1 0.3 --pred2 ../approaches/p_moral-values/output/1_Previous-Sentences-2-Lex-LIWC-22_0.1_Lex-MJD-test.tsv --th2 0.4 --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/self-enh_moral-values/output/comparison-presence-tuned_baseline-tuned_model.txt
# 
# echo "=========="
# echo "=========="
# echo "===== Self-Enh ====="
# echo "=========="
# echo "=========="
# echo ""
# echo "----- Comparing fixed baseline with fixed best model -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/self-enh_moral-values/output/1_Lex-WorryWords_0.23_Baseline-test.tsv --pred2 ../approaches/self-enh_moral-values/output/1_Lex-WorryWords_0.23_Lex-Schwartz-test.tsv | tee ../approaches/self-enh_moral-values/output/comparison-self-enh-fixed_baseline-fixed_model.txt
# echo ""
# echo "----- Comparing fixed baseline with tuned baseline -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/self-enh_moral-values/output/1_Lex-WorryWords_0.23_Baseline-test.tsv --pred2 ../approaches/self-enh_moral-values/output/1_Lex-WorryWords_0.23_Baseline-test.tsv --th2 0.6 --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/self-enh_moral-values/output/comparison-self-enh-fixed_baseline-tuned_baseline.txt
# echo ""
# echo "----- Comparing tuned baseline with tuned model -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/self-enh_moral-values/output/1_Lex-WorryWords_0.23_Baseline-test.tsv --th1 0.6 --pred2 ../approaches/self-enh_moral-values/output/1_Lex-WorryWords_0.23_Lex-Schwartz-test.tsv --th2 0.55 | tee ../approaches/self-enh_moral-values/output/comparison-self-enh-tuned_baseline-tuned_model.txt
#  
# echo "=========="
# echo "=========="
# echo "===== Architecture champions ====="
# echo "=========="
# echo "=========="
# echo ""
# echo "----- Comparing Presence with Direct -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/p_moral-values/output/1_Previous-Sentences-2-Lex-LIWC-22_0.1_Baseline-test.tsv --pred2 ../approaches/moral-values/output/Previous-Sentences-2-test.tsv --th2 0.15 --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/self-enh_moral-values/output/comparison-champions-presence-direct.txt
# echo ""
# echo "----- Comparing Self-Enh with Direct -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/self-enh_moral-values/output/1_Lex-WorryWords_0.23_Baseline-test.tsv --pred2 ../approaches/moral-values/output/Previous-Sentences-2-test.tsv --th2 0.15 --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/self-enh_moral-values/output/comparison-champions-self-enh-direct.txt