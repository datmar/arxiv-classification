# arxiv-classification
Classify arXiv research papers based on topic

## Setup
 - Run `pip install -r requirements.txt`
 - Obtain an api key [here](https://core.ac.uk/api-keys/register) for Core.
 - In config.py file (in root directory) assign API_KEY = "your_api_key"
## Usage
 - Set your desired two topics in get_papers.py
 - Default is set for Deep Learning and Computer Science
 - Run `python get_papers.py`
 - Run `python paper_classifier.py`

## Next Steps
 - Allow user to pass topics through command line arguments
 - Decrease overfitting

NOTE: You can also view the Jupyter Notebook files as an alternative to command line
