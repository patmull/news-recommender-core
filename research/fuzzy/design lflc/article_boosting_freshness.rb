!COMMENT!
Problem solved: Amplifying the impact of fresh news in the article recommendation system



Similarity: U = [0;1]; similarity of pairs of articles based on content-based algorithms; Cosine-Similarity or Jensen-Shannon; (0 = pairs of articles are not similar at all; 1 = pairs of articles identical)


Freshness: V = [0; 336]; freshness of articles; in hours; (0 = article is newer than 1 hour; 336 = approx. half of a month old, its age is no longer meaningful to track)


-----------

Boosting coefficient: W = [0; 1]; Provides the boost based on the similarity and freshness of the article

Fuzzy decomposition of universes for individual variables. The individual fuzzy sets that will form this fuzzy decomposition are then appropriately named. In this example, the decomposition of the universe of variables will always consist of three fuzzy sets:

Similarity: Very high, High, Medium, Low, Very low 
Freshness: Very fresh, fresh, current, slightly old, old
Boosting coefficient: Very high, high, medium, low, very low
!END_COMMENT!

TypeOfDescription=linguistic
InfMethod=Fuzzy_Approximation-functional
DefuzzMethod=SimpleCenterOfGravity
UseFuzzyFilter=false

NumberOfAntecedentVariables=2
NumberOfSuccedentVariables=1
NumberOfRules=1

AntVariable1
 name=similarity
 settings=new
 context=<0,0.5,1>
 discretization=301
End_AntVariable1

AntVariable2
 name=freshness
 settings=new
 context=<0,72,336>
 discretization=301
End_AntVariable2

SucVariable1
 name=boosting
 settings=new
 context=<0,0.5,1>
 discretization=301
End_SucVariable1

RULES
 "" "" | ""
END_RULES
