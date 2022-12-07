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
DefuzzMethod=ModifiedCenterOfGravity
UseFuzzyFilter=false

NumberOfAntecedentVariables=2
NumberOfSuccedentVariables=1
NumberOfRules=16

AntVariable1
 name=similarity
 settings=new
 context=<0,0.5,1>
 discretization=301
 UserTerm
  name=very_low
  type=trapezoid
  parameters= 0 0 0.2 0.4
 End_UserTerm
 UserTerm
  name=low
  type=trapezoid
  parameters= 0.2 0.3 0.4 0.45
 End_UserTerm
 UserTerm
  name=med
  type=trapezoid
  parameters= 0.4 0.45 0.55 0.6
 End_UserTerm
 UserTerm
  name=high
  type=trapezoid
  parameters= 0.7 0.75 0.8 0.898
 End_UserTerm
 UserTerm
  name=very_high
  type=trapezoid
  parameters= 0.8 0.9 1 1
 End_UserTerm
 UserTerm
  name=medium_high
  type=trapezoid
  parameters= 0.55 0.6 0.7 0.75
 End_UserTerm
End_AntVariable1

AntVariable2
 name=freshness
 settings=new
 context=<0,72,336>
 discretization=301
 UserTerm
  name=old
  type=trapezoid
  parameters= 168 192 336 336
 End_UserTerm
 UserTerm
  name=slightly_old
  type=trapezoid
  parameters= 120 144 168 193
 End_UserTerm
 UserTerm
  name=current
  type=trapezoid
  parameters= 72 96 120 144
 End_UserTerm
 UserTerm
  name=fresh
  type=trapezoid
  parameters= 24 48 72 96
 End_UserTerm
 UserTerm
  name=very_fresh
  type=trapezoid
  parameters= 0 0 24 48
 End_UserTerm
End_AntVariable2

SucVariable1
 name=boosting
 settings=new
 context=<0,0.5,1>
 discretization=301
 UserTerm
  name=very_high
  type=trapezoid
  parameters= 0.7 0.8 1 1
 End_UserTerm
 UserTerm
  name=high
  type=trapezoid
  parameters= 0.55 0.6 0.7 0.8
 End_UserTerm
 UserTerm
  name=med
  type=trapezoid
  parameters= 0.4 0.45 0.55 0.6
 End_UserTerm
 UserTerm
  name=low
  type=trapezoid
  parameters= 0.2 0.3 0.4 0.45
 End_UserTerm
 UserTerm
  name=very_low
  type=trapezoid
  parameters= 0 0 0.2 0.3
 End_UserTerm
End_SucVariable1

RULES
 "very_high" "very_fresh or fresh" | "very_high"
 "very_high" "current or slightly_old" | "high"
 "very_high" "old" | "med"
 "high" "very_fresh or fresh" | "high"
 "high" "slightly_old" | "high"
 "high" "current" | "very_high"
 "high" "old" | "med"
 "medium_high" "fresh or current" | "high"
 "medium_high" "slightly_old or old" | "med"
 "medium_high" "very_fresh" | "med"
 "low" "old or slightly_old or current or fresh or very_fresh" | "low"
 "very_low" "old or slightly_old or current or fresh" | "very_low"
 "very_low" "very_fresh" | "low"
 "med" "very_fresh" | "high"
 "med" "old or slightly_old or current" | "med"
 "med" "fresh" | "med"
END_RULES
