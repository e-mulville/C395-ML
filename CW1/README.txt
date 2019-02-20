Imperial College London
CO395 - Introduction to Machine Learning
Coursework 1 - Decision Trees
====================================================
Group ID: 68
Authors:    Alessandro Serena   - as6316@ic.ac.uk
            Enda Mulville       - em1716@ic.ac.uk
            Zaid Aggad          - za716@ic.ac.uk
            David Rovick-Arrojo - der115@ic.ac.uk
====================================================
Readme version: 1.4
Updated on:     11/02/2019



-   To run the script, use:    
        
        python3     cw1.py      [flags]


-   FLAGS
                --visualize     <filename>          --> Print in a nice format to the file specified
                                                        1. cleanTree
                                                        2. cleanTree PRUNED
                                                        3. noisyTree
                                                        4. noisyTree PRUNED

                --inClean       <filename>          --> Use the specified file as Clean dataset

                --inNoisy       <filename>          --> Use the specified file as Clean dataset
                
                -h                                  --> Print README to terminal and terminate program
                --help                              --> Print README to terminal and terminate program



-   NOTE:   multiple flags can be used at the same time

    e.g.    python3  cw1.py  --visualize  out.txt  --inClean  clean.txt
            python3  cw1.py  --inNoisy  noisy.txt  --inClean  clean.txt
