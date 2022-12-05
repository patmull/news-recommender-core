!COMMENT!
Problem solved: This fuzzy approach should add the uncertaity element to coefficient settings of the boosting of various content-based method based on experiences gained from the testing of the system and experimental results.

Besides the adding of the uncertainity element, we also want to model our confidence in the given content-based method itself. For example, in the case of the Word2Vec method we believe in the cosine similarity of the article itself more, however in the case of the Doc2Vec, we believe more in the human manually set coefficient value. We suppose that for some reason, administrator would like to take the advantage of the propertirs of the Doc2Vec results. In that way, it also serves the purpose of some kind of security barrier against the wrong choice of the human-expert.

This variant is dealing with the TF-IDF method which results were the 2nd most reliable of the content-based methods.

Similarity: U = [0;1]; similarity of pairs of articles based on content-based algorithms; Cosine-Similarity or Jensen-Shannon; (0 = pairs of articles are not similar at all; 1 = pairs of articles identical)

Precision settings: V = [0; 10]; multiplying coefficient of the content-based methods set by the administrator of the recommender system.

-----------------

Boosting coefficient: W = [0;10]; Provides the boost based on the similarity and precision setting.

Fuzzy decomposition was this time chosen to fewer number of varibales to keep the level of abstraction elevated, which was the primary reason of the fuzzy approach in the first place.

Fuzzy decomposition of the universes for individual variables:

Similarity: High, Medium, Low
Precision settings: High, Medium, Low
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
 name=freshness_tfidf
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
 context=<0,5,10>
 discretization=301
 UserTerm
  name=med
  type=trapezoid
  parameters= 2 4 6 8
 End_UserTerm
 UserTerm
  name=sml
  type=trapezoid
  parameters= 0 0 2 4
 End_UserTerm
 UserTerm
  name=hig
  type=trapezoid
  parameters= 6 8 10 10
 End_UserTerm
End_SucVariable1

RULES
 "sml" "fresh" | "sml"
 "sml" "current" | "sml"
 "sml" "old" | "med"
 "med" "fresh" | "med"
 "med" "current" | "med"
 "med" "old" | "hig"
 "hig" "fresh" | "med"
 "hig" "current" | "hig"
 "hig" "old" | "hig"
END_RULES
