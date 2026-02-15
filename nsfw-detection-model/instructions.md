# [nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) ➡️ CoreML

<details>
<summary>
TL;DR
</summary>

This folder contains steps to:
1. convert the model
2. compare both models
3. testing in Swift

</details>

<details>
<summary>What model are we using?</summary>

- **Source**: Falconsai/nsfw_image_detection from HuggingFace
- **Input**: expects 224x224 RGB images
- **Output**: binary labelling ("normal" or "NSFW") with an adjustable threshold
- **Deployment Target**: iOS 15+ (targeted in script) 
- **License**: apache-2.0 (yes commercial-use allowed)
- **Designed for**: content moderation and blurring/filtering images for users

</details>


<details>
<summary>Step 1: convert</summary>

1. Download all files for the [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) model and move [here](./prepare-model-for-app/step-1-convert-pytorch-model-to-coreml-model/inputs/)
2. Assuming you have [asdf](https://asdf-vm.com/), setup the environment:
   ```shell
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the script to convert and output the CoreML model [here](./prepare-model-for-app/step-1-convert-pytorch-model-to-coreml-model/outputs/):
   ```shell
   ./convert_model.py
   ```
4. Run the script to compare how both models classify images to see if the conversion was successful:
   ```shell
   ./compare_both_models.py
   ```
   Expecting similar probabilities/low difference, for example:
   ```shell
   ../test-images/normal-example-2.jpg: max_abs_diff=0.000173, pt_top=0, core_top=0
   pt_probs: [9.9982673e-01 1.7325248e-04] ✅ 
   core_probs: [1.000000e+00 1.745224e-04] ✅ 

   ../test-images/nsfw-example-1.jpg: max_abs_diff=0.000302, pt_top=1, core_top=1
   pt_probs: [3.021375e-04 9.996979e-01] 🔞 
   core_probs: [3.0326843e-04 1.0000000e+00] 🔞
   ```
- If you make changes to the script, keep [requirements.txt](./prepare-model-for-app/step-1-convert-pytorch-model-to-coreml-model/requirements.txt) up-to-date:
   ```shell
   pipreqs . --force
   ```
- If you can't run the scripts, grant permission:
   ```shell
   chmod +x convert_model.py
   chmod +x compare_both_models.py
   ```

</details>

<details>
<summary>Step 2: test in Swift</summary>

1. Complete step 1
2. Run the script to see how the same images from step 1 are classified:
   ```shell
   ./test_coreml_model
   ```
   Expecting the same detection results from step 1, for example:
   ```shell
   Testing: normal-example-2.jpg
    - Result: ✅ SAFE (nsfw confidence: 0.02%)

   Testing: nsfw-example-1.jpg
    - Result: 🔞 NSFW (nsfw confidence: 100.00%)
   ```
- If you can't run the script, grant permission:
  ```shell
  chmod +x test_coreml_model.swift
  ```

</details>
