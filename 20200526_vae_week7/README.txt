
Started with a short summary of the theory of the previous week.

Showed with a GUI what an autoencoder learns. This with a VAE trained on the MNIST dataset but also on only the 'two' digits in MNIST.
Showed that autoencoders performs significant dimension reduction (100 fold in our example).
Showed that autoencoders learn very well to reconstruct known data from the domain it was trained on (even on new samples not seen in dataset), but performs badly on data from another domain (anomaly detection).
Experimented with the latent space. Showed that images that look similar to us have similar representations, even though they can differ significantly in pixelspace. Showed linear inerpolations (both in pixel as latent space) between different digits (for example between a 3 and a 5).

Showed practical examples of how unsupervisedly trained models (autoencoders and word vectors) can be used in bioinformatics. 
Including: 
-anomaly detection with red blood cells
-word vectors trained on large reseach text corpus -> simple question answering, finding semantically similar words, and semantic search engine.
-using the columns of MSA to learn the same abstract features to represent them previously published by professor Vriend. 
-using the columns of MSA to learn the same abstract features that represent aminoacids. -> by exploring the latent space showing it indirectly learned about the structure and properties of aa, and the relations between aa.



