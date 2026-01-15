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

# echo "=========="
# echo "=========="
# echo "===== LLM ====="
# echo "=========="
# echo "=========="
# echo ""
# echo "----- Comparing Zero-Shot with Zero-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/llm_moral-values/output/definition/google-gemma-2-9b-it-test.tsv --pred2 ../approaches/llm_moral-values/output/definition/google-gemma-2-9b-it-gated_Presence-test.tsv | tee ../approaches/llm_moral-values/output/comparisons/comparison-zs-direct-sbert.txt
# echo ""
# echo "----- Comparing Zero-Shot with Few-Shot -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/llm_moral-values/output/definition/google-gemma-2-9b-it-test.tsv --pred2 ../approaches/llm_moral-values/output/definition-20/google-gemma-2-9b-it-test.tsv | tee ../approaches/llm_moral-values/output/comparisons/comparison-zs-fs.txt
# echo ""
# echo "----- Comparing Few-Shot with Few-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/llm_moral-values/output/definition-20/google-gemma-2-9b-it-test.tsv --pred2 ../approaches/llm_moral-values/output/definition-20/google-gemma-2-9b-it-gated_Presence-test.tsv | tee ../approaches/llm_moral-values/output/comparisons/comparison-fs-direct-sbert.txt
# echo ""
# echo "----- Comparing QLoRA with best Zero/Few-Shot -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/llm_moral-values/output/qlora-direct-lr-2e-4/google-gemma-2-9b-it-test.tsv --pred2 ../approaches/llm_moral-values/output/definition-20/google-gemma-2-9b-it-gated_Presence-test.tsv | tee ../approaches/llm_moral-values/output/comparisons/comparison-qlora-fs-sbert.txt
# echo ""
# echo "----- Comparing LLM Champion with LLM Ensemble -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/llm_moral-values/output/definition-20/google-gemma-2-9b-it-gated_Presence-test.tsv --pred2 ../approaches/ensembles/llm/all/fixed-hard-champion-test.tsv | tee ../approaches/llm_moral-values/output/comparisons/comparison-llm-champion-ensemble.txt
# echo ""
# echo "----- Comparing Transformers Champion with LLM Champion -----"
# echo ""
# python3 compare_models.py --pred1 ../approaches/ensembles/llm/all/fixed-hard-champion-test.tsv --pred2 ../approaches/moral-values/output/direct_champion-tuned-soft-champion-test.tsv --th2 0.29 | tee ../approaches/llm_moral-values/output/comparisons/comparison-transformers-llm-champions.txt

# echo "=========="
# echo "=========="
# echo "===== LLM - Growth ====="
# echo "=========="
# echo "=========="
# mkdir ../approaches/llm_moral-values/output/comparisons/growth
# echo ""
# echo "----- Comparing Zero-Shot with Zero-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/growth_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/growth_definition/google-gemma-2-9b-it-gated_Growth Anxiety-Free-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/growth/comparison-zs-direct-sbert.txt
# echo ""
# echo "----- Comparing Zero-Shot with Few-Shot -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/growth_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/growth_definition-12/google-gemma-2-9b-it-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/growth/comparison-zs-fs.txt
# echo ""
# echo "----- Comparing Few-Shot with Few-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/growth_definition-12/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/growth_definition-12/google-gemma-2-9b-it-gated_Growth Anxiety-Free-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/growth/comparison-fs-direct-sbert.txt
# echo ""
# echo "----- Comparing best Zero/Few-Shot with QLoRA -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/growth_definition/google-gemma-2-9b-it-gated_Growth Anxiety-Free-test.tsv" --pred2 "../approaches/llm_moral-values/output/growth_qlora/google-gemma-2-9b-it-gated_Growth Anxiety-Free-test.tsv" --prob-cols "Self-direction: thought,Self-direction: action,Stimulation,Hedonism,Achievement,Humility,Benevolence: caring,Benevolence: dependability,Universalism: concern,Universalism: nature,Universalism: tolerance" | tee ../approaches/llm_moral-values/output/comparisons/growth/comparison-zs-qlora-sbert.txt
# echo ""
# echo "----- Comparing LLM Champion with LLM Ensemble -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/growth_definition/google-gemma-2-9b-it-gated_Growth Anxiety-Free-test.tsv" --pred2 "../approaches/ensembles/llm/growth/fixed-hard-champion-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/growth/comparison-llm-champion-ensemble.txt
# echo ""
# echo "----- Comparing Transformers Champion with LLM Champion -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/ensembles/llm/growth/fixed-hard-champion-test.tsv" --pred2 "../approaches/moral-values/output/direct_champion-tuned-soft-champion-test.tsv" --th2 0.29 --prob-cols "Self-direction: thought,Self-direction: action,Stimulation,Hedonism,Achievement,Humility,Benevolence: caring,Benevolence: dependability,Universalism: concern,Universalism: nature,Universalism: tolerance" | tee ../approaches/llm_moral-values/output/comparisons/growth/comparison-transformers-llm-champions.txt

# echo "=========="
# echo "=========="
# echo "===== LLM - Self-Protection ====="
# echo "=========="
# echo "=========="
# mkdir ../approaches/llm_moral-values/output/comparisons/self-protection
# echo ""
# echo "----- Comparing Zero-Shot with Zero-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-protection_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-protection_definition/google-gemma-2-9b-it-gated_Self-Protection Anxiety-Avoidance-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-protection/comparison-zs-direct-sbert.txt
# echo ""
# echo "----- Comparing Zero-Shot with Few-Shot -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-protection_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-protection_definition-11/google-gemma-2-9b-it-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-protection/comparison-zs-fs.txt
# echo ""
# echo "----- Comparing Few-Shot with Few-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-protection_definition-11/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-protection_definition-11/google-gemma-2-9b-it-gated_Self-Protection Anxiety-Avoidance-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-protection/comparison-fs-direct-sbert.txt
# echo ""
# echo "----- Comparing best Zero/Few-Shot with QLoRA -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-protection_definition-11/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-protection_qlora/google-gemma-2-9b-it-gated_Self-Protection Anxiety-Avoidance-test.tsv" --prob-cols "Achievement,Power: dominance,Power: resources,Face,Security: personal,Security: societal,Tradition,Conformity: rules,Conformity: interpersonal,Humility" | tee ../approaches/llm_moral-values/output/comparisons/self-protection/comparison-zsfs-qlora-sbert.txt
# # echo ""
# # echo "----- Comparing LLM Champion with LLM Ensemble -----"
# # echo ""
# # python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-protection_definition/google-gemma-2-9b-it-gated_Self-Protection Anxiety-Avoidance-test.tsv" --pred2 "../approaches/ensembles/llm/self-protection/fixed-hard-champion-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-protection/comparison-llm-champion-ensemble.txt
# echo ""
# echo "----- Comparing Transformers Champion with LLM Champion -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-protection_qlora/google-gemma-2-9b-it-gated_Self-Protection Anxiety-Avoidance-test.tsv" --pred2 "../approaches/moral-values/output/direct_champion-tuned-soft-champion-test.tsv" --th2 0.29 --prob-cols "Achievement,Power: dominance,Power: resources,Face,Security: personal,Security: societal,Tradition,Conformity: rules,Conformity: interpersonal,Humility" | tee ../approaches/llm_moral-values/output/comparisons/self-protection/comparison-transformers-llm-champions.txt
# echo ""
# echo "----- Comparing Transformers/LLM Winner with Ensemble of both -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/moral-values/output/direct_champion-tuned-soft-champion-test.tsv" --th1 0.29 --pred2 "../approaches/ensembles/llm/self-protection-transformers/tuned-hard-champion-test.tsv" --prob-cols "Achievement,Power: dominance,Power: resources,Face,Security: personal,Security: societal,Tradition,Conformity: rules,Conformity: interpersonal,Humility" | tee ../approaches/llm_moral-values/output/comparisons/self-protection/comparison-transformers-llm-champion-ensemble-both.txt

# echo "=========="
# echo "=========="
# echo "===== LLM - Social Focus ====="
# echo "=========="
# echo "=========="
# mkdir ../approaches/llm_moral-values/output/comparisons/social-focus
# echo ""
# echo "----- Comparing Zero-Shot with Zero-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/social-focus_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/social-focus_definition/google-gemma-2-9b-it-gated_Social Focus-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/social-focus/comparison-zs-direct-sbert.txt
# echo ""
# echo "----- Comparing Zero-Shot with Few-Shot -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/social-focus_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/social-focus_definition-11/google-gemma-2-9b-it-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/social-focus/comparison-zs-fs.txt
# echo ""
# echo "----- Comparing Few-Shot with Few-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/social-focus_definition-11/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/social-focus_definition-11/google-gemma-2-9b-it-gated_Social Focus-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/social-focus/comparison-fs-direct-sbert.txt
# echo ""
# echo "----- Comparing best Zero/Few-Shot with QLoRA -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/social-focus_definition-11/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/social-focus_qlora/google-gemma-2-9b-it-gated_Social Focus-test.tsv" --prob-cols "Security: societal,Tradition,Conformity: rules,Conformity: interpersonal,Humility,Benevolence: caring,Benevolence: dependability,Universalism: concern,Universalism: nature,Universalism: tolerance" | tee ../approaches/llm_moral-values/output/comparisons/social-focus/comparison-zsfs-qlora-sbert.txt
# echo ""
# echo "----- Comparing LLM Champion with LLM Ensemble -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/social-focus_definition-11/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/ensembles/llm/social-focus/fixed-hard-champion-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/social-focus/comparison-llm-champion-ensemble.txt
# echo ""
# echo "----- Comparing Transformers Champion with LLM Champion -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/ensembles/llm/social-focus/fixed-hard-champion-test.tsv" --pred2 "../approaches/moral-values/output/Baseline-test.tsv" --th2 0.3 --prob-cols "Security: societal,Tradition,Conformity: rules,Conformity: interpersonal,Humility,Benevolence: caring,Benevolence: dependability,Universalism: concern,Universalism: nature,Universalism: tolerance" | tee ../approaches/llm_moral-values/output/comparisons/social-focus/comparison-transformers-llm-champions.txt
# # echo ""
# # echo "----- Comparing Transformers/LLM Winner with Ensemble of both -----"
# # echo ""
# # python3 compare_models.py --pred1 "../approaches/moral-values/output/direct_champion-tuned-soft-champion-test.tsv" --th1 0.29 --pred2 "../approaches/ensembles/llm/social-focus-transformers/tuned-hard-champion-test.tsv" --prob-cols "Security: societal,Tradition,Conformity: rules,Conformity: interpersonal,Humility,Benevolence: caring,Benevolence: dependability,Universalism: concern,Universalism: nature,Universalism: tolerance" | tee ../approaches/llm_moral-values/output/comparisons/social-focus/comparison-transformers-llm-champion-ensemble-both.txt

# echo "=========="
# echo "=========="
# echo "===== LLM - Personal Focus ====="
# echo "=========="
# echo "=========="
# mkdir ../approaches/llm_moral-values/output/comparisons/personal-focus
# echo ""
# echo "----- Comparing Zero-Shot with Zero-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/personal-focus_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/personal-focus_definition/google-gemma-2-9b-it-gated_Personal Focus-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/personal-focus/comparison-zs-direct-sbert.txt
# echo ""
# echo "----- Comparing Zero-Shot with Few-Shot -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/personal-focus_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/personal-focus_definition-10/google-gemma-2-9b-it-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/personal-focus/comparison-zs-fs.txt
# echo ""
# echo "----- Comparing Few-Shot with Few-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/personal-focus_definition-10/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/personal-focus_definition-10/google-gemma-2-9b-it-gated_Personal Focus-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/personal-focus/comparison-fs-direct-sbert.txt
# echo ""
# echo "----- Comparing best Zero/Few-Shot with QLoRA -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/personal-focus_definition-10/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/personal-focus_qlora/google-gemma-2-9b-it-gated_Personal Focus-test.tsv" --prob-cols "Self-direction: thought,Self-direction: action,Stimulation,Hedonism,Achievement,Power: dominance,Power: resources,Face,Security: personal" | tee ../approaches/llm_moral-values/output/comparisons/personal-focus/comparison-zsfs-qlora-sbert.txt
# # echo ""
# # echo "----- Comparing LLM Champion with LLM Ensemble -----"
# # echo ""
# # python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/personal-focus_definition-10/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/ensembles/llm/personal-focus/fixed-hard-champion-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/personal-focus/comparison-llm-champion-ensemble.txt
# echo ""
# echo "----- Comparing Transformers Champion with LLM Champion -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/personal-focus_definition-10/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/moral-values/output/direct_champion-tuned-soft-champion-test.tsv" --th2 0.29 --prob-cols "Self-direction: thought,Self-direction: action,Stimulation,Hedonism,Achievement,Power: dominance,Power: resources,Face,Security: personal" | tee ../approaches/llm_moral-values/output/comparisons/personal-focus/comparison-transformers-llm-champions.txt
# echo ""
# echo "----- Comparing Transformers/LLM Winner with Ensemble of both -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/moral-values/output/direct_champion-tuned-soft-champion-test.tsv" --th1 0.29 --pred2 "../approaches/ensembles/llm/personal-focus-transformers/tuned-hard-champion-test.tsv" --prob-cols "Self-direction: thought,Self-direction: action,Stimulation,Hedonism,Achievement,Power: dominance,Power: resources,Face,Security: personal" | tee ../approaches/llm_moral-values/output/comparisons/personal-focus/comparison-transformers-llm-champion-ensemble-both.txt

# echo "=========="
# echo "=========="
# echo "===== LLM - Openness ====="
# echo "=========="
# echo "=========="
# mkdir ../approaches/llm_moral-values/output/comparisons/openness
# echo ""
# echo "----- Comparing Zero-Shot with Zero-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/openness_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/openness_definition/google-gemma-2-9b-it-gated_Openness to Change-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/openness/comparison-zs-direct-sbert.txt
# echo ""
# echo "----- Comparing Zero-Shot with Few-Shot -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/openness_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/openness_definition-5/google-gemma-2-9b-it-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/openness/comparison-zs-fs.txt
# echo ""
# echo "----- Comparing Few-Shot with Few-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/openness_definition-5/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/openness_definition-5/google-gemma-2-9b-it-gated_Openness to Change-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/openness/comparison-fs-direct-sbert.txt
# echo ""
# echo "----- Comparing best Zero/Few-Shot with QLoRA -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/openness_definition-5/google-gemma-2-9b-it-gated_Openness to Change-test.tsv" --pred2 "../approaches/llm_moral-values/output/openness_qlora/google-gemma-2-9b-it-gated_Openness to Change-test.tsv" --prob-cols "Self-direction: thought,Self-direction: action,Stimulation,Hedonism" | tee ../approaches/llm_moral-values/output/comparisons/openness/comparison-zsfs-qlora-sbert.txt
# # echo ""
# # echo "----- Comparing LLM Champion with LLM Ensemble -----"
# # echo ""
# # python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/openness_definition-5/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/ensembles/llm/openness/fixed-hard-champion-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/openness/comparison-llm-champion-ensemble.txt
# echo ""
# echo "----- Comparing Transformers Champion with LLM Champion -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/openness_definition-5/google-gemma-2-9b-it-gated_Openness to Change-test.tsv" --pred2 "../approaches/moral-values/output/Baseline-test.tsv" --th2 0.3 --prob-cols "Self-direction: thought,Self-direction: action,Stimulation,Hedonism" | tee ../approaches/llm_moral-values/output/comparisons/openness/comparison-transformers-llm-champions.txt
# echo ""
# echo "----- Comparing Transformers/LLM Winner with Ensemble of both -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/moral-values/output/Baseline-test.tsv" --th1 0.3 --pred2 "../approaches/ensembles/llm/openness-transformers/tuned-soft-champion-test.tsv" --th2 0.17 --prob-cols "Self-direction: thought,Self-direction: action,Stimulation,Hedonism" | tee ../approaches/llm_moral-values/output/comparisons/openness/comparison-transformers-llm-champion-ensemble-both.txt

# echo "=========="
# echo "=========="
# echo "===== LLM - Conservation ====="
# echo "=========="
# echo "=========="
# mkdir ../approaches/llm_moral-values/output/comparisons/conservation
# echo ""
# echo "----- Comparing Zero-Shot with Zero-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/conservation_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/conservation_definition/google-gemma-2-9b-it-gated_Conservation-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/conservation/comparison-zs-direct-sbert.txt
# echo ""
# echo "----- Comparing Zero-Shot with Few-Shot -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/conservation_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/conservation_definition-8/google-gemma-2-9b-it-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/conservation/comparison-zs-fs.txt
# echo ""
# echo "----- Comparing Few-Shot with Few-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/conservation_definition-8/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/conservation_definition-8/google-gemma-2-9b-it-gated_Conservation-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/conservation/comparison-fs-direct-sbert.txt
# echo ""
# echo "----- Comparing best Zero/Few-Shot with QLoRA -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/conservation_definition-8/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/conservation_qlora/google-gemma-2-9b-it-gated_Conservation-test.tsv" --prob-cols "Face,Security: personal,Security: societal,Tradition,Conformity: rules,Conformity: interpersonal,Humility" | tee ../approaches/llm_moral-values/output/comparisons/conservation/comparison-zsfs-qlora-sbert.txt
# # echo ""
# # echo "----- Comparing LLM Champion with LLM Ensemble -----"
# # echo ""
# # python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/conservation_definition-8/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/ensembles/llm/conservation/fixed-hard-champion-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/conservation/comparison-llm-champion-ensemble.txt
# echo ""
# echo "----- Comparing Transformers Champion with LLM Champion -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/conservation_qlora/google-gemma-2-9b-it-gated_Conservation-test.tsv" --pred2 "../approaches/moral-values/output/Baseline-test.tsv" --th2 0.3 --prob-cols "Face,Security: personal,Security: societal,Tradition,Conformity: rules,Conformity: interpersonal,Humility" | tee ../approaches/llm_moral-values/output/comparisons/conservation/comparison-transformers-llm-champions.txt
# echo ""
# echo "----- Comparing Transformers/LLM Winner with Ensemble of both -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/moral-values/output/Baseline-test.tsv" --th1 0.3 --pred2 "../approaches/ensembles/llm/conservation-transformers/tuned-hard-champion-test.tsv" --prob-cols "Face,Security: personal,Security: societal,Tradition,Conformity: rules,Conformity: interpersonal,Humility" | tee ../approaches/llm_moral-values/output/comparisons/conservation/comparison-transformers-llm-champion-ensemble-both.txt

# echo "=========="
# echo "=========="
# echo "===== LLM - Self-Transcendence ====="
# echo "=========="
# echo "=========="
# mkdir ../approaches/llm_moral-values/output/comparisons/self-transcendence
# echo ""
# echo "----- Comparing Zero-Shot with Zero-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-transcendence_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-transcendence_definition/google-gemma-2-9b-it-gated_Self-Transcendence-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-transcendence/comparison-zs-direct-sbert.txt
# echo ""
# echo "----- Comparing Zero-Shot with Few-Shot -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-transcendence_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-transcendence_definition-7/google-gemma-2-9b-it-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-transcendence/comparison-zs-fs.txt
# echo ""
# echo "----- Comparing Few-Shot with Few-Shot hier (SBERT gate) -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-transcendence_definition-7/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-transcendence_definition-7/google-gemma-2-9b-it-gated_Self-Transcendence-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-transcendence/comparison-fs-direct-sbert.txt
# echo ""
# echo "----- Comparing best Zero/Few-Shot with QLoRA -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-transcendence_definition-7/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-transcendence_qlora/google-gemma-2-9b-it-gated_Self-Transcendence-test.tsv" --prob-cols "Humility,Benevolence: caring,Benevolence: dependability,Universalism: concern,Universalism: nature,Universalism: tolerance" | tee ../approaches/llm_moral-values/output/comparisons/self-transcendence/comparison-zsfs-qlora-sbert.txt
# echo ""
# echo "----- Comparing LLM Champion with LLM Ensemble -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-transcendence_definition-7/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/ensembles/llm/self-transcendence/fixed-hard-champion-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-transcendence/comparison-llm-champion-ensemble.txt
# echo ""
# echo "----- Comparing Transformers Champion with LLM Champion -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-transcendence_definition-7/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/moral-values/output/Baseline-test.tsv" --th2 0.3 --prob-cols "Humility,Benevolence: caring,Benevolence: dependability,Universalism: concern,Universalism: nature,Universalism: tolerance" | tee ../approaches/llm_moral-values/output/comparisons/self-transcendence/comparison-transformers-llm-champions.txt
# # echo ""
# # echo "----- Comparing Transformers/LLM Winner with Ensemble of both -----"
# # echo ""
# # python3 compare_models.py --pred1 "../approaches/moral-values/output/Baseline-test.tsv" --th1 0.3 --pred2 "../approaches/ensembles/llm/self-transcendence-transformers/tuned-hard-champion-test.tsv" --prob-cols "Humility,Benevolence: caring,Benevolence: dependability,Universalism: concern,Universalism: nature,Universalism: tolerance" | tee ../approaches/llm_moral-values/output/comparisons/self-transcendence/comparison-transformers-llm-champion-ensemble-both.txt

echo "=========="
echo "=========="
echo "===== LLM - Self-Enhancement ====="
echo "=========="
echo "=========="
mkdir ../approaches/llm_moral-values/output/comparisons/self-enhancement
echo ""
echo "----- Comparing Zero-Shot with Zero-Shot hier (SBERT gate) -----"
echo ""
python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-enhancement_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-enhancement_definition/google-gemma-2-9b-it-gated_Self-Enhancement-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-enhancement/comparison-zs-direct-sbert.txt
echo ""
echo "----- Comparing Zero-Shot with Few-Shot -----"
echo ""
python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-enhancement_definition/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-enhancement_definition-6/google-gemma-2-9b-it-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-enhancement/comparison-zs-fs.txt
echo ""
echo "----- Comparing Few-Shot with Few-Shot hier (SBERT gate) -----"
echo ""
python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-enhancement_definition-6/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-enhancement_definition-6/google-gemma-2-9b-it-gated_Self-Enhancement-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-enhancement/comparison-fs-direct-sbert.txt
echo ""
echo "----- Comparing best Zero/Few-Shot with QLoRA -----"
echo ""
python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-enhancement_definition-6/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/llm_moral-values/output/self-enhancement_qlora/google-gemma-2-9b-it-gated_Self-Enhancement-test.tsv" --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/llm_moral-values/output/comparisons/self-enhancement/comparison-zsfs-qlora-sbert.txt
# echo ""
# echo "----- Comparing LLM Champion with LLM Ensemble -----"
# echo ""
# python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-enhancement_definition-6/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/ensembles/llm/self-enhancement/fixed-hard-champion-test.tsv" | tee ../approaches/llm_moral-values/output/comparisons/self-enhancement/comparison-llm-champion-ensemble.txt
echo ""
echo "----- Comparing Transformers Champion with LLM Champion -----"
echo ""
python3 compare_models.py --pred1 "../approaches/llm_moral-values/output/self-enhancement_definition-6/google-gemma-2-9b-it-test.tsv" --pred2 "../approaches/moral-values/output/Previous-Sentences-2-test.tsv" --th2 0.15 --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/llm_moral-values/output/comparisons/self-enhancement/comparison-transformers-llm-champions.txt
echo ""
echo "----- Comparing Transformers/LLM Winner with Ensemble of both -----"
echo ""
python3 compare_models.py --pred1 "../approaches/moral-values/output/Previous-Sentences-2-test.tsv" --th1 0.15 --pred2 "../approaches/ensembles/llm/self-enhancement-transformers/tuned-hard-champion-test.tsv" --prob-cols "Hedonism,Achievement,Power: dominance,Power: resources,Face" | tee ../approaches/llm_moral-values/output/comparisons/self-enhancement/comparison-transformers-llm-champion-ensemble-both.txt