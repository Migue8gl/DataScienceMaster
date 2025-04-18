- #hypothesis | #statistics | #usecases

# Tests para muestras independientes
- Las filas han de ser independientes entre sí. Los grupos han de ser independientes entre sí.
    ## Test para dos muestras
    - **_Paramétrico_**. Los residuos han de ser normales. Dependiendo de si las varianzas son similares o no (según nos indique el test de **Levene**) tenemos:
            - _Varianzas similares_: **t-test**
            - _Varianzas distintas_: **Welch**
    - **_No paramétrico_**: **Mann-Whitney** (es un test sobre las medianas cuando las distribuciones son similares y es un test de dominancia estocástica en otro caso)
    ## _Test para k muestras_
    - **_Paramétrico_**. Los residuos han de ser normales.
            - _Varianzas similares_: **One-way Anova**. Post-hoc: **Tukey HSD**
            - _Varianzas distintas_: **Welch**. Post-hoc: **Games-Howell**
            - En cualquiera de los casos anteriores se puede aplicar un post-hoc genérico como por ejemplo Holm.
	- **_No paramétrico_**: **Kruskal-Wallis** (es un test sobre las medianas cuando las distribuciones son similares y es un test de dominancia estocástica en otro caso). Post-hoc: **Dwass-Steel-Critchlow-Fligner**, **Dunn**, **Conover**, **Nemenyi**. También se puede aplicar un post-hoc genérico.
# Tests para muestras dependientes (medidas repetidas)
- Las filas han de ser independientes entre sí.
    ## Test para dos muestras
    - **_Paramétrico_**. Los residuos de las diferencias han de ser normales. Se aplica el **paired t-test**
    - **_No paramétrico_**: Si la diferencia de las dos variables es una distribución simétrica, se aplicará **Wilcoxon**. En otro caso, el **sign-test**
    ## Test para k muestras
    - **_Paramétrico_**. Los residuos han de ser normales. Se aplica el **Anova de medidas repetidas**. Si no se verifica la hipótesis de esfericidad (test de **Mauchly**) se aplica la corrección de **GreenHouse-Geiser**. Post-hoc con **Tukey** o uno genérico.
    - **_No paramétrico_**: **Friedman** (es un test sobre las medianas cuando las distribuciones son similares y es un test de dominancia estocástica en otro caso). Post-hoc: **Durbin-Conover**, **Nemenyi**, **Eisinga**, etc. También se puede aplicar un post-hoc genérico.