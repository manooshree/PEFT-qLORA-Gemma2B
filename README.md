# PEFT-qLORA-Gemma2B
**Supervised fine-tuning on a Python code instruction dataset of Google's open-source Gemma-2B model via qLoRA to create your own "Copilot"**
Datset: https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style/tree/c5191ac0d60716fdbb94500d9fdd40c58f2e6ed3
<img width="1023" alt="Screenshot 2024-03-05 at 10 38 05 AM" src="https://github.com/manooshree/PEFT-qLORA-Gemma2B/assets/25752491/e41fda26-9f0c-4e65-8f32-32c574a3c263">

**Key**: Memory efficient model loading done by double-quantizing weights and storing in 4-bit float format (Dettmers et. al). Essentially, weights are quantized and then weight coefficients are quantized as well.
<img width="573" alt="Screenshot 2024-03-05 at 10 39 04 AM" src="https://github.com/manooshree/PEFT-qLORA-Gemma2B/assets/25752491/e72cc215-868c-4860-8eb8-db857d198ad1">

LoRA Basics: To make fine-tuning more efficient, LoRA’s approach is to represent the weight updates with two smaller matrices (called update matrices) through low-rank decomposition. These new matrices can be trained to adapt to the new data while keeping the overall number of changes low. The original weight matrix remains frozen and doesn’t receive any further adjustments. To produce the final results, both the original and the adapted weights are combined.

We are reducing dimensions of the weight matrix from Φ to θ with θ << Φ. Hu et al. shows there is little performance reduction with LoRA technique but huge time and memory gains (Hu et al.). The added LoRA matrix is initialized with two matrices: A, B. B is a matrix of 0's and A is initialized by normally distributed noise with hyperparameter σ.
![image](https://github.com/manooshree/PEFT-qLORA-Gemma2B/assets/25752491/53cb6b96-734f-419d-9c0f-222bda95ac95)

<img width="620" alt="Screenshot 2024-03-05 at 10 41 00 AM" src="https://github.com/manooshree/PEFT-qLORA-Gemma2B/assets/25752491/de351d27-dc53-464b-b37c-d570b79043ab">

Example Quantization on MLP in Transformer architecture
<img width="1120" alt="Screenshot 2024-03-05 at 10 44 38 AM" src="https://github.com/manooshree/PEFT-qLORA-Gemma2B/assets/25752491/d7220851-58d4-45df-8c74-c2807fb75865">


Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2024). Qlora: Efficient finetuning of quantized llms. Advances in Neural Information Processing Systems, 36.
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

