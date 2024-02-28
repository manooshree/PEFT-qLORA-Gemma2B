# PEFT-qLORA-Gemma2B
Supervised fine-tuning of Google's open-source Gemma-2B model to optimize writing Python code

![image](https://github.com/manooshree/PEFT-qLORA-Gemma2B/assets/25752491/53cb6b96-734f-419d-9c0f-222bda95ac95)

PEFT is a library from Hugging Face which comes with several options to train models efficiently, one of them is LoRA.

To make fine-tuning more efficient, LoRA’s approach is to represent the weight updates with two smaller matrices (called update matrices) through low-rank decomposition. These new matrices can be trained to adapt to the new data while keeping the overall number of changes low. The original weight matrix remains frozen and doesn’t receive any further adjustments. To produce the final results, both the original and the adapted weights are combined.

Relevant Paper: https://arxiv.org/pdf/2106.09685.pdf
