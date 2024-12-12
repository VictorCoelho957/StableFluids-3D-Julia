# Simulação de Fluidos Estáveis em 3D
[![NPM](https://img.shields.io/npm/l/react)]() 

Este repositório contém uma implementação em Julia para a simulação de fluidos estáveis em 3D utilizando a Transformada Rápida de Fourier (FFT). Este trabalho foi baseado no repositório original do autor Ceyron (https://github.com/Ceyron), disponível em https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/stable_fluids_fft_3d.jl.

# Sobre o Projeto

Esta implementação tem como objetivo explorar conceitos de computação gráfica e física computacional, com foco na resolução das equações de Navier-Stokes para fluidos incompressíveis em 3D. As principais equações tratadas são:

Equação do Momentum



# Condição de Incompressibilidade


# Modificações Realizadas

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

ParaView (para visualização dos resultados)

Como Executar

Clone o repositório:

git clone https://github.com/seu-usuario/seu-repositorio.git

Instale as dependências necessárias no ambiente Julia:

using Pkg
Pkg.add("FFTW")
Pkg.add("Plots")

Execute o script principal:

julia stable_fluids_fft_3d.jl

Utilize o ParaView para abrir os arquivos gerados e visualizar os resultados da simulação.

# Créditos

Implementação original: Ceyron (https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/stable_fluids_fft_3d.jl)

Alterações e adaptações: Victor Coelho e André Moura

Proposto na disciplina de Computação Gráfica pelo Professor Doutor Haroldo Gomes Barroso Filho

# Contato dos Autores


## André Moura

GitHub: André Moura

LinkedIn: André Moura

E-mail Institucional: 

## Victor Coelho

GitHub: Victor Coelho

LinkedIn: Victor Coelho

E-mail Institucional:

# Licença

Consulte a licença no repositório original para informações sobre os termos de uso..
