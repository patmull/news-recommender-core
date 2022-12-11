!COMMENT!
Problem solved: This fuzzy approach should add the uncertaity element to a boosting coefficients of the based on the precision of various content-based method based on the experiences gained from the testing of the system and the experiments.

We want to model our confidence in the given content-based method itself. For example, in the case of the Word2Vec method, we believe in the cosine similarity of the article itself more, however in the case of the Doc2Vec, from our experimental results we understand that the cosine similarity was not that precise in providing similar results to a given article, thus we want to rather take advantage of the freshness of the article since news portal but also blogs usually tend to push their latest articles higher than the old ones (see the percentage of hwo the iDNES articles on the front-page were old).

This variant is dealing with the Word2Vec method (trained on iDNES with the whole body test) which results were the most reliable of the content-based methods.

Similarity: U = [0;1]; similarity of pairs of articles based on content-based algorithms; Cosine-Similarity
Freshness: V = [0; 5]; number of days of how the article is old

-----------------

Boosting coefficient: W = [0;1]; Provides the boost based on the similarity and freshness of the article

Fuzzy decomposition was this time chosen to fewer number of varibales to keep the level of abstraction elevated, which was the primary reason of the fuzzy approach in the first place.

Fuzzy decomposition of the universes for individual variables:

Similarity: High, Medium, Low
Freshness: Fresh, Current, Old
Boosting CB Mixing coefficient: High, Medium, Low
!END_COMMENT!

TypeOfDescription=linguistic
InfMethod=Fuzzy_Approximation-functional
DefuzzMethod=ModifiedCenterOfGravity
UseFuzzyFilter=false

NumberOfAntecedentVariables=2
NumberOfSuccedentVariables=1
NumberOfRules=9

AntVariable1
 name=similarity
 settings=new
 context=<0,0.5,1>
 discretization=301
 UserTerm
  name=med
  type=trapezoid
  parameters= 0.2 0.4 0.6 0.8
 End_UserTerm
 UserTerm
  name=sml
  type=trapezoid
  parameters= 0 0 0.2 0.4
 End_UserTerm
 UserTerm
  name=hig
  type=trapezoid
  parameters= 0.6 0.8 1 1
 End_UserTerm
End_AntVariable1

AntVariable2
 name=freshness
 settings=new
 context=<0,2,5>
 discretization=301
 UserTerm
  name=current
  type=trapezoid
  parameters= 1 2 3 4
 End_UserTerm
 UserTerm
  name=fresh
  type=trapezoid
  parameters= 0 0 1 2
 End_UserTerm
 UserTerm
  name=old
  type=trapezoid
  parameters= 3 4 5 5
 End_UserTerm
End_AntVariable2

SucVariable1
 name=final_coefficient
 settings=new
 context=<0,0.5,1>
 discretization=301
 UserTerm
  name=med
  type=trapezoid
  parameters= 0.2 0.4 0.6 0.8
 End_UserTerm
 UserTerm
  name=sml
  type=trapezoid
  parameters= 0 0 0.2 0.4
 End_UserTerm
 UserTerm
  name=hig
  type=trapezoid
  parameters= 0.6 0.8 1 1
 End_UserTerm
End_SucVariable1

RULES
 "sml" "fresh" | "sml"
 "sml" "current" | "sml"
 "sml" "old" | "sml"
 "med" "fresh" | "med"
 "med" "current" | "med"
 "med" "old" | "med"
 "hig" "fresh" | "med"
 "hig" "current" | "hig"
 "hig" "old" | "hig"
END_RULES
