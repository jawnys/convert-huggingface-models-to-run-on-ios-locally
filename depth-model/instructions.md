# [Depth Anything V2](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2) ➡️ CoreML


<details>
<summary>
TL;DR
</summary>

This folder contains steps to:
1. convert the model
2. compare both models, testing [`shirt.png`](./shirt.png)
3. using Swift to test [`shirt.png`](./shirt.png)

</details>

<details>
<summary>
What model are we using?
</summary>

- **Source**: DepthAnything/Depth-Anything-V2 on GitHub
- **Output**: depth PNG for your input image[^1]
- **Deployment Target**: iOS 15+ (targeted in script) 
- **License**: apache-2.0 (yes commercial-use allowed)

[^1]: Depth values are **relative depth**, not absolute metric distance in meters.

</details>


<details>
<summary>
Quick start
</summary>

1. Download the model from [here](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#pre-trained-models), choose `Depth-Anything-V2-Small` as it has an [Apache-2.0 license](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#license)
2. Copy the model into [step-1-convert/inputs](./step-1-convert/inputs)
3. Run step 1

   ```shell
   cd step-1-convert
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ./convert_model.py
   ./compare_both_models.py
   ```
4. Compare the 2 test results:
   * [shirt-depth-coreml.png](./step-1-convert/outputs/shirt-depth-coreml.png)
   * [shirt-depth-coreml.png](./step-1-convert/outputs/shirt-depth-coreml.png)
5. Run step 2
   
   ```shell
   cd ../step-2-test-coreml-model
   ./test_coreml_model.swift
   ```  
   If you can't run the script, grant permission:
   ```shell
   chmod +x test_coreml_model.swift
   ```
6. Use the CoreML model from [step-1-convert/outputs](./step-1-convert/outputs) in your project

</details>
