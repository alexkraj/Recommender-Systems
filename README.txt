============================== HOW TO RUN =============================

This will detail how to run the system on a linux-based machine. If
using Windows, please use a git-bash terminal. 

1. Ensure that all the correct imports are installed in the environment,
namely flask, numpy, pandas, scipy.sparse.linalg. All these imports
can also be found at the top of musicrecc.py. Python 3.7.1 was used in
development. 

2. From inside the main directory, run the following command
        $python3 musicrecc.py
   
   You will expect an output similar to this:
        * Serving Flask app "musicrecc" (lazy loading)
        * Environment: production
          WARNING: This is a development server. Do not use it in a 
	  production deployment.
          Use a production WSGI server instead.
        * Debug mode: on
        * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    
    Follow the specified IP address to access the single-page web app.
    The initial start up might take a few seconds.The GET and POST 
    requests are logged in the terminal from which the python script 
    is run. 


=========================== ABOUT THE SYSTEM ==========================

The Flask server reads in 2 .cvs files: 
1. data_music.cvs - 4 columns structured as follows: 
    (Song_ID, Artist, Title, Genre)
    ~140 entries
2.data_ratings.cvs - 4 columns structured as follows:
    (User_ID, Song_ID, Rating, Landscape)
    ~500 entries
This dataset was obtained from the following source:
@inproceedings{inproceedings,
author = {Baltrunas, Linas and Kaminskas, Marius and Ludwig, Bernd and Moling, Omar and Ricci, Francesco and Aydin, Aykan and Lüke, Karl-Heinz and Schwaiger, Roland},
year = {2011},
month = {08},
pages = {89-100},
title = {InCarMusic: Context-Aware Music Recommendations in a Car},
Volume = {85},
journal = {Lect. Note. Bus. Info. Proc.},
doi = {10.1007/978-3-642-23014-1_8}
}

The system generates recommendations using Matrix Factorization via 
Singular Value Decomposition.

The user is defaulted to user 6. One can choose any other user profile
by changing the variable 'user' in musicrecc.py.

*note: some of the books have foreign characters that were not preserved 
in their names/Authors and thus appear strangely on the webapp. 