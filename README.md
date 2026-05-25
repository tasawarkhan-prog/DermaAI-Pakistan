# ⚕️ DermaAI Pakistan v2.0

[![Live Demo](https://img.shields.io/badge/HuggingFace-Live%20Demo-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Tasawar-prog1/dermaai-pakistan-v2)

DermaAI Pakistan is an advanced, bilingual (English/Urdu) AI-powered dermatology assistant. It utilizes a **Multi-Agent RAG (Retrieval-Augmented Generation) architecture** to analyze skin conditions with explainable AI insights and Pakistan-specific medical knowledge.

## 🚀 Features
- **Multi-Provider AI:** Switch between **Google Gemini, Groq Cloud (Llama), and Alibaba Qwen** vision models.
- **Bilingual Support:** Full English and Urdu (اردو) localization for local patient accessibility.
- **RAG Engine:** Uses FAISS and Sentence Transformers to retrieve clinically relevant information from a curated knowledge base.
- **XAI Heatmaps:** Implements Grad-CAM and gradient-based attention maps to visualize where the AI is focusing on the skin lesion.
- **Local Medicine DB:** Integrated with Pakistan-specific DRAP formulary data.

## 🛠 Tech Stack
- **Frontend:** Streamlit
- **AI/LLMs:** Google Generative AI SDK, OpenAI-compatible API (Groq/Qwen)
- **Computer Vision:** PyTorch, EfficientNet-B0, torchvision
- **Explainability:** pytorch-grad-cam, OpenCV
- **Vector Search:** FAISS, Sentence Transformers (all-MiniLM-L6-v2)
- **Deployment:** Hugging Face Spaces

## 🏗 Architecture
The system employs a sophisticated multi-agent approach:
1.  **Vision Agent:** Analyzes dermatological images for condition detection.
2.  **RAG Agent:** Retrieves evidence-based clinical context.
3.  **Medicine Agent:** Maps detections to local brand-name medications.
4.  **Grad-CAM Agent:** Generates attention heatmaps for visual diagnostic support.
5.  **Chat Agent:** Multi-turn context-aware assistant for follow-up questions.

## 📈 Model Training
The project includes a dedicated training pipeline (`train_colab.py`) built to fine-tune an **EfficientNet-B0** model on the HAM10000 dataset, enabling custom diagnostic features that are then deployed in production.

## 🎯 Use Cases
- **Patient Self-Assessment:** Initial skin condition screening
- **Medical Education:** Understanding dermatological conditions
- **Accessibility:** Urdu-language healthcare AI for underserved populations
- **Research:** Explainable AI in healthcare

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Tasawar Abbas Khan**
- Email: khantasawar68@gmail.com
- LinkedIn: [Tasawar Abbas Khan](https://www.linkedin.com/in/tasawar-abbas-khan-niazi-b87328314)
- GitHub: [@tasawarkhan-prog](https://github.com/tasawarkhan-prog)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests for improvements.

## ⚠️ Disclaimer
DermaAI Pakistan is an **educational and informational tool only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed, board-certified dermatologist for medical concerns.

---

**Building trustworthy, explainable AI solutions for healthcare in Pakistan.** ✨

*Developed as a contribution to AI-driven healthcare accessibility.*
