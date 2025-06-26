Technical Case 1

This tool automatically updates market capitalization values in quarterly review PowerPoint presentations by extracting relevant data from financial Excel spreadsheets. The app is Dockerized and runs fully offline once started.

1. Prerequisites
Install Docker
On Windows
Download Docker Desktop:
https://www.docker.com/products/docker-desktop

Verify installation in PowerShell:
docker --version
docker run hello-world

On Linux i.e. Ubuntu
Run the following in your terminal:
sudo apt remove docker docker-engine docker.io containerd runc
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker

Verify installation:
docker run hello-world

2. Launch the App

Once Docker is installed, run the following commands:
docker load -i update_mkt_cap.tar
docker run -p 8501:8501 update-mkt-cap
Access the application at http://localhost:8501 or, if running on a VM, at the external IP shown in your console.

3. Application Usage
Upload the required files:

PowerPoint file i.e., CRC_Top_Positions_4Q24.pptx
Excel file/s containing financial data i.e., capitalization tables

The application processes the data, updates the market cap values in the presentation, and provides a modified .pptx file for download.

Market cap values are extracted from specific sheets i.e. Simple Model Case using the headers labeled Capitalization.

4. Technical Notes
The app currently supports a specific naming and file structure. For other financial quarters, additional logic may be required to match tickers and periods based on filename or sheet content.

For demonstration purposes, each market cap value in the output .pptx is increased by $99 to clearly indicate modification.

To improve generalizability:
Ensure the PowerPoint uses "Company Ticker" rather than "Company Name".
Ensure the Excel sheets include ticker symbols aligned with the PowerPoint file.

5. Cleanup 
To free disk space after you're done:
docker system prune -a --volumes




Technical Case 2

This tool allows you to query expert call transcripts using a lightweight, fully offline RAG pipeline. It supports uploading PDFs of expert calls and generating answers grounded in the uploaded content. The context used to generate the answer is retained and observable with a click of a button.

1. Load the Docker Image
In your terminal, run:
docker load -i expert_query_tool.tar

2. Launch the Application
docker run -p 8501:8501 expert-query-tool

3. Access the App in Your Browser
Local machine:
Open http://localhost:8501

Remote/VM instance:
Replace localhost with the external IP address shown in your console (e.g., http://<external-ip>:8501).

Key Features
Document Upload
Upload expert call transcripts in PDF format.

RAG-Based Question Answering
Ask questions about the content. Responses are concise and grounded in the retrieved context from transcripts.

Example Questions
Here are some effective sample questions to ask after uploading expert call documents:
What did the expert say about Uber's margins in Q1?
What did the expert say about Uber's driver incentives in Q1?
Was there any mention of regulatory risks?
How did the expert assess Uber’s competitive positioning?
Do you think Uber will partner with Waymo?

Offline Capable
No internet connection is required once the Docker image is loaded and running.

Improvements
Since pretrained models were used, many other models can be tested. Fine-tuning is also an option. This allows the model to specialize in a domain appropriate area. 
As for guarding against hallucinations, one of several measures was taken. Ensure that the answer is rooted in the context. An analysis can be done and integrated into the model to prevent out of bounds responses. 



