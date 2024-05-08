# RAG - README

## Resumo
O **Retriever-Augmented Generative** (RAG) é uma arquitetura de modelo de linguagem que combina capacidades de recuperação de documentos (Retriever) com capacidades de geração de texto (Generative) para tarefas de questionamento e resposta. Ele é especialmente útil em situações em que se precisa responder perguntas com base em um contexto mais amplo, como conversas anteriores.

## Sobre o código
Este código implementa um sistema RAG utilizando a biblioteca _**LangChain**_ para integrar funcionalidades de recuperação e geração de texto. Aqui está uma breve visão geral do que o código faz:

* Carrega documentos da web relacionados ao CrewAI.
* Carrega exemplos de código relacionados ao CrewAI de um diretório local.
* Divide os documentos de texto em pedaços menores para processamento eficiente.
* Divide os exemplos de código Python em pedaços menores para processamento eficiente.
* Gera embeddings de texto usando Ollama, um modelo de linguagem.
* Cria um banco de vetores (Chroma) a partir dos documentos de texto e código.
* Configura um sistema de geração de texto usando RAG.
* Configura um sistema de recuperação de documentos com histórico de conversa.
* Combina os sistemas de recuperação e geração para formar uma cadeia de processamento para responder a perguntas com base em um contexto histórico.

Para executar este código, é necessário ter as seguintes dependências instaladas:

* **`langchain_community`** para carregar documentos e exemplos de código.
* **`langchain_text_splitters`** para dividir texto em pedaços menores.
* **`langchain_community.embeddings`** para gerar embeddings de texto.
* **`langchain_community.vectorstores`** para criar e persistir o banco de vetores.
* **`langchain_community.chat_models`** para configurar o modelo de conversação.
* **`langchain_core`** para manipulação de mensagens e cadeias de processamento.

Além disso, é necessário definir variáveis de ambiente para os caminhos dos arquivos e modelos necessários. Este código pode ser usado como base para construir sistemas de questionamento e resposta mais avançados, especialmente em ambientes onde a contextualização e o histórico de conversa desempenham um papel importante.
