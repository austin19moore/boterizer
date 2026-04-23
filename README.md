# boterizer

## Setup

1. **Install Python** on your system ([Python](https://www.python.org/downloads/))

2. **Clone this repository** to your local machine

3. **Export Discord chat history** in JSON format using the Discord chat exporter ([DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter))  
   Place the exported file in the top level directory

4. **Install required packages:**
   ```
   pip install torch torchvision unsloth unsloth--zoo
   ```

5. **Clone the base model** you want to finetune into the top of this directory from HuggingFace:
   - Recommended 0.8B: [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B)
   - Recommended 4B: [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B)

6. **Update configuration** in the top of `train_model.py`:
   - `CSV_PATH`
   - `MODEL_PATH`
   - `USERNAME`

7. **Run the training script:**
   ```
   python train_model.py
   ```

8. **Set up a Discord bot** and obtain the API key: [Discord Developer Portal](https://discord.com/developers/applications)

9. **Run the Discord bot** (included):
   
   **Option A - Using Docker:**
   ```
   docker-compose up
   ```
   
   **Option B - Run locally:**
   ```
   npm run start
   ```
   
   > Change the variables at the top of `./discordBot/index.ts`, or update them in the `docker-compose.yml` file.