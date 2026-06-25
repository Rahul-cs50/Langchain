from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import grandalf
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
llm2 = HuggingFaceEndpoint(
    model="Qwen/Qwen3-8B",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

)

model = ChatHuggingFace(llm=llm)
model2= ChatHuggingFace(llm=llm2)

parser = StrOutputParser()
prompt1 = PromptTemplate(
    template="create notes on the topic being provided: {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="create short quizs on the topic being provided: {topic}",
    input_variables=["text"]
)
prompt3 = PromptTemplate(
    template="create notes and quizes for the following information provided : {notes} and the quiz: {quiz}",
    input_variables=["notes","quiz"]
)

paralell_chain = RunnableParallel({
    'notes':prompt1|model|parser,
    'quiz': prompt2|model2|parser
})
mergered_Chain = prompt3|model|parser

chain = paralell_chain|mergered_Chain



text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({'topic':text})

print(result)

chain.get_graph().print_ascii()