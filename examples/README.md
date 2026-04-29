# Examples

## `query.py` — Single image query

```bash
python query.py \
  --url https://your-anchor.run.app \
  --adapter open_circuit \
  --image path/to/image.jpg \
  --prompt "Is there an open circuit defect? Answer YES or NO."
```

Output:
```
Answer: YES  (216ms)
```

## `batch_query.py` — Run a directory of images

```bash
python batch_query.py \
  --url https://your-anchor.run.app \
  --adapter open_circuit \
  --images ./test_images/ \
  --prompt "Defect present? YES or NO."
```

Output:
```
Querying 24 images with adapter 'open_circuit'
Image                                    Answer        ms
------------------------------------------------------------
img_001.jpg                              YES          218ms
img_002.jpg                              NO           214ms
img_003.jpg                              YES          221ms
```

## `langchain_integration.py` — Use Anchor as a LangChain tool

```python
from langchain_integration import AnchorVisionTool

tool = AnchorVisionTool(
    endpoint="https://your-anchor.run.app",
    adapter="open_circuit",
    prompt="Is there a defect? Answer YES or NO.",
)

result = tool.invoke({"image_path": "image.jpg"})
# → "YES"
```

Drop into any LangChain agent or chain as a vision inspection tool.

---

## `finetune.py` — Train your own LoRA adapter

Train a LoRA adapter on your own dataset, then load it into Anchor.

**Step 1: Prepare dataset (CSV)**

```csv
image_path,label
images/ok_01.jpg,NO
images/defect_01.jpg,YES
images/defect_02.jpg,YES
```

**Step 2: Train**

```bash
pip install transformers peft accelerate torch Pillow

python finetune.py \
  --model google/paligemma2-3b-pt-448 \
  --data ./dataset.csv \
  --output ./my_adapter \
  --task "Is there a defect? Answer YES or NO." \
  --epochs 3 \
  --lora-rank 16
```

**Step 3: Load into Anchor**

```bash
# Copy adapter to lora directory
cp -r ./my_adapter /lora/my_adapter/

# Rebuild Docker image (or restart with new LORA_PATH)
# Then query:
python query.py \
  --url http://localhost:8080 \
  --adapter my_adapter \
  --image test.jpg \
  --prompt "Is there a defect? Answer YES or NO."
```
