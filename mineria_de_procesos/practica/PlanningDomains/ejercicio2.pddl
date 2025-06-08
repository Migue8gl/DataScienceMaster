(define (domain zeno-travel)


(:requirements
  :typing
  :fluents
  :derived-predicates
  :negative-preconditions
  :universal-preconditions
  :disjuntive-preconditions
  :conditional-effects
  :htn-expansion

  ; Requisitos adicionales para el manejo del tiempo
  :durative-actions
  :metatags
 )

(:types aircraft person city - object)
(:constants slow fast - object)
(:predicates (at ?x - (either person aircraft) ?c - city)
             (in ?p - person ?a - aircraft)
             (different ?x ?y) ;;(igual ?x ?y)
             (hay-fuel ?a ?c1 ?c2)
             )
(:functions (fuel ?a - aircraft)
            (distance ?c1 - city ?c2 - city)
            (slow-speed ?a - aircraft)
            (fast-speed ?a - aircraft)
            (slow-burn ?a - aircraft)
            (fast-burn ?a - aircraft)
            (capacity ?a - aircraft)
            (refuel-rate ?a - aircraft)
            (total-fuel-used)
            (boarding-time)
            (debarking-time)
            (igual ?x ?y) {return (?x == ?y)}
            )

;; el consecuente "vac�o" se representa como "()" y significa "siempre verdad"
 (:derived
   (igual ?x ?x) ())

(:derived 
  (different ?x ?y)
  (not (igual ?x ?y))
)




;; este literal derivado se utiliza para deducir, a partir de la información en el estado actual, 
;; si hay fuel suficiente para que el avión ?a vuele de la ciudad ?c1 a la ?c2
;; el antecedente de este literal derivado comprueba si el fuel actual de ?a es mayor que 1. 
;; En este caso es una forma de describir que no hay restricciones de fuel. Pueden introducirse una
;; restricción más copleja  si en lugar de 1 se representa una expresión más elaborada (esto es objeto de
;; los siguientes ejercicios).
(:derived
 (hay-fuel ?a - aircraft ?c1 - city ?c2 - city)
 (>= (fuel ?a) (* (distance ?c1 ?c2) (slow-burn ?a))))

(:task transport-person
 :parameters (?p - person ?c - city)

 (:method Case1
  :precondition (at ?p ?c)
  :tasks ()
 )

 (:method Case2
  :precondition (and (at ?p ?c1)
                     (at ?a ?c2)
                     (igual ?c1 ?c2))
  :tasks (
    (board ?p ?a ?c1)
    (mover-avion ?a ?c1 ?c)
    (debark ?p ?a ?c)
  )
 )

 (:method Case3
  :precondition (and (at ?p ?c1)
                     (at ?a ?c2)
                     (different ?c1 ?c2))
  :tasks (
    (mover-avion ?a ?c2 ?c1)
    (transport-person ?p ?c)
  )
 )
)

(:task mover-avion
  :parameters (?a - aircraft ?c1 - city ?c2 - city)

  ;; Caso 1: hay suficiente fuel para hacer el viaje
  (:method fuel-suficiente
    :precondition (hay-fuel ?a ?c1 ?c2)
    :tasks (
      (fly ?a ?c1 ?c2)
    )
  )

  ;; Caso 2: no hay suficiente fuel, se necesita repostar primero
  (:method repostar-y-volar
    :precondition (not (hay-fuel ?a ?c1 ?c2))
    :tasks (
      (refuel ?a ?c1)
      (mover-avion ?a ?c1 ?c2)
    )
  )

  
)
 
(:import "Primitivas-ZenoTravel.pddl") 


)