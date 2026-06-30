# BERT Chat Classifier — Flask Prototype

Flask 2.2 implementation of the binary BERT classification chatbot described in the dissertation.

## Run locally

**1. Open this folder in VS Code**

**2. Create a virtual environment** (Terminal)
```bash
cd flask_prototype
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the server**
```bash
python app.py
```

**5. Open the app**

Go to `http://localhost:5000` in your browser.

---

## Add your fine-tuned BERT model

1. Copy your saved model folder (containing `config.json`, `pytorch_model.bin` or `model.safetensors`, `tokenizer_config.json`, `vocab.txt`) into `flask_prototype/model/`

2. Uncomment the PyTorch and Transformers lines in `requirements.txt`:
   ```
   torch>=2.0.0
   transformers>=4.30.0
   ```

3. Re-run `pip install -r requirements.txt`

4. In `app.py`, replace the body of `classify_message()` with the BERT inference code shown in the comments inside that function.

5. Restart the server — it will now use your actual fine-tuned model.
