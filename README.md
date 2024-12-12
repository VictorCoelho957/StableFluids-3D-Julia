Simulação de Fluidos Estáveis em 3D

Este repositório contém uma implementação em Julia para a simulação de fluidos estáveis em 3D utilizando a Transformada Rápida de Fourier (FFT). Este trabalho foi baseado no repositório original do autor Ceyron, disponível em english/simulation_scripts/stable_fluids_fft_3d.jl.

Sobre o Projeto

Esta implementação tem como objetivo explorar conceitos de computação gráfica e física computacional, com foco na resolução das equações de Navier-Stokes para fluidos incompressíveis em 3D. As principais equações tratadas são:

Equação do Momentum



Condição de Incompressibilidade



Onde:

: campo de velocidade do fluido.

: tempo.

: densidade do fluido.

: pressão.

: viscosidade cinemática.

: forças externas aplicadas ao fluido.

Modificações Realizadas

As alterações foram feitas por Victor Coelho e André Moura, como parte da disciplina de Computação Gráfica no curso de Engenharia da Computação, ministrada pelo Professor Doutor Haroldo Gomes Barroso Filho. As mudanças incluem:

Adaptação do código para inclusão de novos parâmetros gráficos.

Ajustes na visualização para melhorar a compreensão dos fenômenos simulados.

Comentários adicionais e refatoração para facilitar o uso educacional.

Dependências

Certifique-se de que as seguintes dependências estão instaladas para executar o código:

Julia

FFTW.jl

Plots.jl

LinearAlgebra (incluído na biblioteca padrão do Julia)

Como Executar

Clone o repositório:

git clone https://github.com/seu-usuario/seu-repositorio.git

Instale as dependências necessárias no ambiente Julia:

using Pkg
Pkg.add("FFTW")
Pkg.add("Plots")

Execute o script principal:

julia stable_fluids_fft_3d.jl

Créditos

Implementação original: Ceyron (Repositório)

Alterações e adaptações: Victor Coelho e André Moura

Proposto na disciplina de Computação Gráfica pelo Professor Doutor Haroldo Gomes Barroso Filho

Licença

Consulte a licença no repositório original para informações sobre os termos de uso.
