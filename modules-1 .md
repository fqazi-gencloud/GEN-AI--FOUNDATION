Notable Breakthroughs in generative AI
Timeline of additional developments in generative AI


Natural Language Processing
A major breakthrough in generative AI and in particular, the development of natural processing language (NLP) was the introduction of Generative Pre-trained Transformer (or GPT) models. In 2018, the first version of a GPT model was introduced by the California-based research organization OpenAI. This neural network model was designed to generate text, engage in human-like conversation, and perform a number of language-based tasks. Its development marked a turning point in the widespread use of machine learning that is prevalent today. This transformer technology can now be used to automate and refine lots of tasks, including text translation, writing promotional materials, business reports, and software programs. As with most models, the value of this type of model is a combination of fast GPU processors, large amounts of training data, and the ability to operate at scale. GPT is trained through a self-supervised approach on mountains of text, documents, and other data gathered across the Internet. The model performs a language modeling task, meaning it predicts the next word given the context that was previously provided. In 2023, the latest version (GPT 4) was released, capable of generating tens of thousands of words, making it a significant improvement over earlier versions.
self-supervised: a machine learning technique where a model trains itself to learn one part of the input from another part of the input. It is also known as predictive or pretext learning.

Recurrent Neural Network
A recurrent neural network (or RNN) is a type of artificial neural network that is well-suited for processing sequential data. Sequential data means data where the order matters, like text, speech, or video.

RNNs have something called memory which allows them to remember information about previous inputs. This gives RNNs the ability to take context into account when processing data. For example, if you wanted to predict the next word in a sentence, it helps to know the previous words.

Some key points about RNNs:

They have loops which allow information to persist across time steps. This gives them memory.

They can process input sequences of varying lengths. The length does not need to be fixed ahead of time.

They are useful for tasks like language translation, speech recognition, and text generation.

Examples of RNN architectures include LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit). These were designed to improve on basic RNNs.

Generative Adversarial Network
A generative adversarial network (or GAN) is a type of generative AI model that is able to create new, synthetic data that resembles real data. GANs consist of two neural networks - a generator and a discriminator - that compete against each other to become better at their respective tasks.

The generator's job is to create fake data that looks as realistic as possible. It starts by creating random noise and slowly learns to convert that noise into data that mimics real data (like images, text, etc).

The discriminator's job is to look at data and judge whether it is real or fake. It receives real data as well as fake data from the generator. By looking at these inputs, it learns to get better at distinguishing real data from fake.

The two networks play a cat and mouse game - the generator tries to fool the discriminator by creating better fakes, while the discriminator tries to catch the fakes. This process repeats over and over as both networks improve. The end result is a generator that can create highly realistic synthetic data.

GANs are useful for generating new data in domains where data is scarce, like medical imaging. They are also used for image editing applications, creating artificial profiles, and more. The key is that the GAN learns to mimic real data without being explicitly programmed for the task.
Text-to-Image Models
Beginning around 2020, additional AI engines began to surface and create breakthroughs that advanced beyond text, into more complex features like creating life-like images. After the release of its large language model, OpenAI released DALL-E in 2021, a text-to-image model capable of producing high-resolution imagery, and other image-based tasks. As a transformer model modified to 'swap text for pixels'[1] DALL-E can also generate alternative versions of an image based on textual input. For example, the model can create an image of a dog, then further modify the image (based on specific inputs) to locate the dog in different settings like a meadow, a beach,  or a farm. It can even outfit the dog into  an elaborate costume, or a pair of eyeglasses. Creative imagery aside, the important aspect is that a model pretrained on large datasets full of text and images is capable of generating endless amounts of output based on many types of inputs. Three neural networks that makes this possible are:

CLIP (Contrastive Language-Image Pre-training) which is responsible for recognizing text and creating a sketch of the future image;

GLIDE which is responsible for converting the sketch into a final low-resolution image;

A third neural network responsible for enhancing the resolution of the generated image and adding additional details where required.


Stable Diffusion
Released in 2022, stable diffusion is an open-source neural network developed by 
Stability.ai
 that allows images of art quality to be created in a short amount of time. This model can also assist with making precision edits or alterations to an original image. Stable Diffusion is developed using a diffusion model that generates images from random noise. It is trained to remove unnecessary parts from a sample image through a series of iterations, producing a result after a few stages of processing. This algorithm works by adding noise to the original image and gradually transforming it into photorealistic images or art. 

Another generative AI tool for image creation is the Midjourney neural network, also created in 2022. Midjourney is a subscription-based product powered by image-generating algorithms users can access through a chatbot interface. It recognizes prompts, and converts the text into life-like images. Similar to Stable Diffusion, Midjourney creates images from noise. It primary benefits is that users are not required to be proficient with programming languages, meaning anyone is capable of creating high quality images. Midjourney can produce output so detailed that some experts have trouble distinguishing work produced by the tool from real or original productions.

AI-enabled apps like Chat GPT, and Speechify represent the rapid growth generative AI has experienced over the past several years. As new models for driving new services continue to emerge, the need to improve performance and output quality must also improve. In this module, we will discuss a few design, performance, and evaluation considerations for large language models.

Large language models take months to train and require massive storage, compute resources, many hours, and energy consumption. There are costs associated with each of these, and once you add it all up, models quickly become expensive to build and operate. This creates many challenges for the specialized hardware, data collection and build techniques that have to be coordinated to build a quality model. With the right planning, complexity and cost can be reduced. However, additional factors still need to be addressed. Bias and integrity, among others, are a problem because LLMs are trained on human language. As you can imagine, dialogue often lacks integrity or is filled with ethical issues, including false statements and character misrepresentations based on race, gender, or religious beliefs. For today's popular models, user access is on the rise. We must maintain awareness and be prepared to mitigate any drawbacks. Here's a list of some key criteria that should be considered when designing a large language model. Data. The data used for training models is crucial. It should be high quality, diverse, and representative of the topics and tasks the model will be expected to handle. More data is typically better as it provides additional context for learning. However, great care should be taken to ensure that the data comes from reliable sources. Model architecture. The structure and number of parameters contained within a model should be suitable for performing all tasks. Parameters define characteristics like the amount of embeddings and tention and weights applied during development and training. Like knobs on a machine, parameters are adjusted to properly optimize performance. Large language models contain hundreds of billions of parameters, with some models being developed exceeding the trillion parameter mark. Bigger does not always mean better, as trade offs must be made between the size, training time, computational cost, and complexity. Large language models for medicine and health care should be designed specifically for that use case and may exclude tasks like creative writing or audio generation that aren't relevant to patient care. Training methods. The techniques used to train a model must consider factors like transfer, learning from existing models, data accuracy and calibration for continuous improvement. Models that suffer from a lack of training often miss important patterns in the data set which can carry over to the newly generated data. This creates the potential for hallucination or details that seem realistic but are not grounded in the actual training data. For example, an image model may hallucinate a tree in the background of an ocean scene even if no images of trees in the ocean was included as part of the training data. The risk of hallucination increases as generative models become more powerful and advanced at producing realistic content. False details often go unnoticed and is problematic when output needs to accurately reflect the real world. A large language model could hallucinate false information or inaccurate quotes. The effects could be damaging if it involves areas like news reporting or legal reviews in a court case. Earlier, we mentioned transfer learning from existing models. One approach that involves transferred learning is fine tuning, which is a process that extends the capabilities of a pre trained model with new data or tasks. Rather than starting with a new model, fine tuning customizes an existing model for a desired use case, making the bill process more efficient. Another benefit of fine tuning is that it incrementally improves model performance on the new dataset while maintaining the general capabilities learned during pre training. Overall, knowledge transfer and fine tuning helps create more specialized generative AI tools and services. Safety and ethics. Careful thought should be put into how to make models safe, avoid harmful biases, and ensure ethical AI practices are followed. Techniques like human oversight, information filtering, and parameter adjustments that adhere to best practices help. Model design should make personal privacy and confidentiality a priority by avoiding details like an individual's address of residence, phone number, medical history, and other sensitive information. Evaluation. Rigorous testing of diverse datasets is needed to understand model capabilities and limitations. Automatic metrics and human evaluation should be used. Users of generative AI applications should be allowed to provide feedback for model operators to review and make adjustments accordingly. Deployment. Real world requirements like latency, compute cost, and model size should be considered when determining optimal designs for a production system.
: Added to Selection.


Design Factors for Large Language Models
Data Collection & Preparation
We have routinely discussed the need for high-quality data. Even if you have a well-designed architecture, and training procedures, data quality remains an important factor. Data ultimately determines how well a model produces output that is realistic, accurate, diverse, and properly suits the use case. Gathering and preparing data for a model is a rigorous task, and the process must address issues like, 

How large is the data?

Where will it be stored, and how will storage impact overall costs?

Is the data secure, and free from unauthorized access that threatens its integrity?

Does the data adhere to industry standards like PII or personal identifiable information?

Only after all factors have been addressed can the process of data preparation begin. Here are some basic steps to follow, 

Data Collection: The first step is to obtain text data for large language model training. Data can be gathered from online books, websites like Wikipedia, social media channels, or from converted audio transcripts with web crawlers that pull the information regularly. Datasets can also be downloaded from open source repositories, or purchased from providers that sell customized information for industries like finance, healthcare, and aviation. 

Data Storage: Once the raw data is collected, it needs to be kept in a location that is safe and durable. Storage vendors, and cloud providers offer flexible and highly durable services like databases and distributed file systems for efficient retrieval and processing of the data. 

Preprocessing: Preprocessing data for a large language model involves:

Cleaning: Removing irrelevant content like HTML tags or unwanted symbols and formats.

Tokenization - Splitting the text into individual words or tokens.

Normalization - Converting text to the same case (e.g. lowercasing) and handling spelling variations.

Removing 'stop' words - Removing words like "a", "and", "the" that generally don't provide much meaning.

The preprocessed data is then used to build examples that associate the input text with target outputs the model is trained to generate.

Training
With data now ready, the creation process can begin. Let's take a high-level look at a process flow for developing a new model. This example represents just one of many different methods that could be implemented,

Finalize the model architecture: The model starts out as a blank slate. Here is where the number and types of neural network layers is determined, and how they are connected. For large language model architectures, this step configures the transformer to be used.

Initialize the model parameters: In the beginning, a model has no intelligence. The embedding, self-attention weights and parameter values mentioned earlier are randomly initialized to small values at this point.

Input text fragments into the model: Training data is split into small pieces, and then fed into the model one piece at a time.

Back-propagation: This is the key step where the model learns. The model makes a prediction on each piece of input. It then compares the predictions to the actual text and calculates an error value. It then propagates the error backwards to update the parameters across all layers.  At this stage, the input and back-propagation steps are repeated over hundreds of thousands of times. With each iteration, the model gets better at understanding the patterns and predicting the next words. This is the core of the model training process. 

Evaluation
Evaluation & Tuning: During training, model performance is evaluated on a validation dataset. Additional parameters like learning rate may be adjusted to improve the output results. Quality evaluation serves several key purposes. 

First, evaluation measures progress as we train models longer or change their design. 

Second, evaluation allows model variants to be compared to determine which works best for certain tasks. 

Finally, evaluation can benchmark generative AI against human performance.   As AI capabilities evolve, understanding their strengths helps us determine how to best leverage both types of intelligences.



Below is an introductory list of terms to describe the components that make up Generative AI. These will help you get started, and help with future lessons. As more terms are introduced, we will define them as well.

fine tuning — The process of taking an existing large language model and training it further on a smaller dataset that is specific to a certain task. For example, you could fine tune the Claude LLM on a dataset of customer support tickets to adapt it to generate responses suited for customer service.

hallucination — refers to when a generative AI model creates content that is not actually present in the input data it was trained on. This can happen because it was trained to produce realistic outputs, but lacks a strong mechanism to ensure the outputs precisely match the training data.

sentiment analysis — the use of natural language processing (NLP) techniques to determine the emotional tone or attitude expressed in a piece of text. The goal of sentiment analysis is to classify text as positive, negative, or neutral.

text summarization — the process of taking a long piece of text and generating a shorter version that captures the key points. Large language models can be trained to perform text summarization automatically. A large language model is shown many examples of texts paired with human-written summaries during training. By analyzing these examples, the model learns to identify the most important information in the original text and condense it into a summary. 

tokens — a basic unit of data used by models. When you provide a text prompt to a model, the text is broken down into smaller segments representing a word or portion of a word. Tokens are building blocks used by the model to understand the meaning of the text and generate a response. The model is trained on huge datasets of text, so it learns the patterns of how these word tokens fit together to make coherent sentences and passages. When generating new text, the model looks at the input tokens you provide, recognizes patterns based on its training, and uses probabilities to predict the most likely next tokens to produce a relevant and human-like response.

weights — numeric values that represent the strength of the connections in a neural network, which are tuned by training and allow the model to make intelligent predictions. The overall set of weights makes up what the model has learned about language.


taken from discussion -courseria 



Emerging technologies have always disrupted society in positive and negative ways. With generative AI's rapid rise, things like creative art journalism, education, and the basic concept of truth, some would argue is being put to the test. The use of AI powered chat bots, photo realistic image generators, and other tools are raising the stakes for addressing the ethical dilemmas for what can be produced using generative AI. Let's watch a brief interview that further discusses the ethical concerns with generative AI. Joining me for this discussion on ethics and generative AI is Jen Looper who leads the academic advocacy team within developer relations at AWS. Jen is an author and someone who is well versed on this topic, having delivered a number of talks on ethical AI at universities and conferences around the globe. Welcome Jen. Thank you. I'm delighted to be here to discuss this really important topic. It's great to have you here. Briefly tell us why this topic is such an important area of focus, not only for businesses and society, but for you personally? Sure. As someone with an educational background in the humanities, it's become clear to me over the past year or so that while rapid advances in AI have made many mundane tasks much easier to accomplish, there's a risk that we lose sight of our own agency as creative thinking creatures by becoming a little bit too dependent on it. So for this reason, the creation and use of these tools I think, needs to be couched in an environment infused with awareness of ethical concerns. Building Go vocabulary and muscle memory when thinking about AI will serve us as both builders and consumers. For those taking this course, what are some of the key points you hope that they will walk away with from this discussion? Right, I would love folks to be able to identify eight major issues that need to be addressed to ethically build and use generative AI, and I'd like students to visualize themselves as active participants in an era where they're invested in both the creation and the proper usage of these tools. Well, thank you again for joining us to share some valuable insights. Let's get right into a few questions. There's this well known saying that goes with great power, come to great responsibility. Give us your take of the power inherent within generative AI and the role that we as responsible builders must play. For sure. We're in this era where this great power has been given to all people with Internet connections worldwide who can access tools like ChatGPT. Fortunately, there are many industry groups who are tackling the question of proper building and usage, such as the AI ethics principles by Stanford's Institute for Human Centered AI, or the AI ethics guidelines by the OECD. And that's the Organization for Economic Cooperation and Development. It's an economics organization. But many of these frameworks boil down to encompass eight major considerations, so let's go over those. The first is controlability, and that's the ability to watch, monitor, and guide AI systems. There's also privacy and security that's obtaining and using data and models appropriately. There's safety stopping any harmful outcomes that might be produced. There's fairness, watching out for unfair impacts on different groups of people. Veracity and robustness, making sure that outputs are actually correct. Explainability, making sure that we can actually understand how outputs are generated. There's also transparency and that's ensuring that people can make good choices about when and how they use an AI system, and finally, governance, infusing the AI supply chain with models built with good practices in mind. Each one of these topics could be an entire course. But in general, the ethical considerations put great importance on where the data is sourced to train large language models or LLMs such that original creators are not harmed. Also how the usage of these LLM should be attributed so that humans are placed firmly in the center of AI output. You brought up chatGPT, we've all seen tools like this and how it has changed the trajectory for how we all get things done and that's certainly the case for students wanting to complete assignments. Generative AI tools as a whole are raising the stakes with their ability to complete complex task in a matter of seconds. These things can write a resume, they can generate computer code, they can even draft an essay for lots of topics. Now, while these are all amazing innovations, they are also problematic when you consider the impact they're having on how we actually learn. My next question for you is with students relying on AI to complete their work, what can professors and administrators do to maintain academic integrity in the classroom? Oh, this is a hot one. As an educator by training, I've watched schools and universities struggle to manage the rapid evolution that's been forced upon classrooms. But almost every sector of human activity either has been or probably will be impacted by these tools. It's really in our interest to think about how we need to evolve to use these tools responsibly and change the way we do our day to day activities if necessary. For example, in the computer science classroom, new tooling embedded in IDEs lets us discover details about the license around the code that's being suggested. That really offers teaching moments around various software licenses and the proper usage of them. The careful use of AI in the classroom in fact allows us to get very crisp on attribution, primary sources, licenses, and copyright if we teach intentionally. Okay, so using the tools ethically is important in doing so to help tackle a challenging problem is in a lot of ways acceptable. Let's now turn our attention to the creation of the models. Model creation, for the most part, is beyond the end user's control, meaning the responsibility lies with the creator. What can we learn about the ethical considerations around building the models we're relying on? Well, LLMs are just that. They're large language models and they don't appear out of thin air. Let's talk about three important aspects of building these models. Their energy consumption, their explainability, and their fairness. Hucking face. Researchers actually calculated the carbon used in training the Bloom model using a software tool called Code Carbon which tracked the carbon dioxide emission that models training was producing in real time over a period of 18 days. And they estimated that Bloom's training led to 25 metric tons. And that's the equivalent of 60 New York to London flights of carbon dioxide emissions. Training for similar models are estimated to emit between 75 and 500 metric tons of carbon dioxide. That's really a lot of emissions. Yeah. Yeah. Clearly a lot of energy being used and definitely something we need to address. You also talked about explainability and fairness that points to the data within the model. So what about that? Yeah, well first there's the question of copyright. And legal scholars are really wrestling with who owns this novel generated data. We often don't really even know whether the data used in training was copyright or if you can copyright generated data, how can you cited this generated data? How can you check accuracy? What happens to the whole concept of verifiable primary sources as we seek to build explainable models? Second, we really ought to consider the diversity of the data. These models can ingest only the data that's been digitized and is public right care needs to be taken to use a diverse data set. Since gaps in data or incomplete or incorrect data can lead to bias in models and impact their fairness. Finally, training these models involve scraping enormous amounts of data from the public internet. There's a lot of negative and even dangerous data, and humans have been hired to detox or remove it. It's been reported that some companies outsource this cleaning task to low paid workers who suffer the mental consequences of labeling disturbing images and text. So I just want to underline that. It's on us to pay attention to the energy and the human resources required to build these models and to mitigate harms caused as we try to create explainable and fair AI models. Yeah, it sounds like we have a lot to think about as we begin to learn and understand more about building and using models. To wrap us up here, what's our major takeaway as we enter this developing world of generative AI? Well, I think that we need to make sure to remember always that we humans are and will be for the foreseeable future and control of this technology, honestly. It's up to us to build it and use it responsibly. Keep your creative, critical mind sharp, and make sure you know who is behind these AI models, what their business models and incentives are. Think about any kind of guard rails that you can align to or even create yourself to ensure your safety. And finally, have faith in yourself and your own humanity to know that you're one of the amazing humans who created such things in the first place. Outstanding. Thank you so much for your wonderful insights and for joining us today, Jen.
en
​

Well, that wraps up our course. I want to thank you for taking part in this learning journey and certainly hope that you found the material we presented useful and enlightening. We covered several key topics and introduced lots of terms, starting with a brief history of artificial intelligence, followed by core concepts in machine learning, deep learning, large language models, and much more. Congratulations on completing the course. I hope you learned a lot about generative AI and will continue to learn even more through additional study. You should feel more empowered to talk about generative AI with friends and colleagues. And for those who are still not familiar with generative AI, invite them to take this course as well. One final thought, as you strive to become a more avid user or builder of generative AI applications, I ask that you do so in a manner that is responsible, ethical, and will benefit us all.




















