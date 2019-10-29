Question Statement:     
Given N abstracts, and M research papers, calculate the similarity of the abstracts to the research papers and output the results in a similarity matrix.     

Solution:       

Preprocessing
Preprocessing of raw data using BeautifulSoup and NLTK
Removing White Spaces, Stop words, and stemming the input, and then tokenizing
it.

Keywords:
Using NLP based learning/methods for generating keywords for the input documents.
Selecting the keywords depending on an alterable threshold frequency.

Model:
Using LDA, and keywords, treating the abstracts as the unknowns, and training
LDA on the input full articles and the obtained keywords. Jenson Shannon
divergence distance is used in order to obtain similarities.

Output:
Obtained output is in form of a 2D similarity matrix, with values ranging from
0(No similarity) to 1(Perfect matching).    


Dependencies:
nltk
gensim
LdaModel
scipy
bs4
sys
codecs

Instructions to run code - 
1. Install given dependencies.
2. run script.py arg1 arg2 where arg1 is abstracts.csv and arg2 is papers.csv
3. The output is stored in similarity_matrix.csv 
