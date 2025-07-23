PREFIX_PROMPT = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response_A}

###Response B:
{response_B}

###Reference Answer:
{reference_answer}

###Score Rubric:
{rubric}

###Feedback: """



instruction_cap = "The ground truth captions of a provided image are listed here. They are separated by a semi-colon. Please carefully observe the image and come up with a caption for the image."
rubric_cap = "Is the answer well correlated with the ground truth captions?"

rubric_mmbench = "The rubric is two-fold: 1. Does the option chosen (A or B or C or D) match the reference answer? 2. Does the explanation provided justify the choice made?"
