from collections import Counter
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO, Entrez, AlignIO, Phylo
from Bio.Blast import NCBIXML
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceCalculator


# Análise e Manipulação de Sequências


def transformar_seq(seq: Seq, tabela_trad: str = "Bacterial") -> None:
    
    """
    Aplica várias transformações a uma sequência nucleotídica.

    Esta função gera e apresenta:
    - Sequência original
    - Sequência complementar
    - Sequência complementar reversa
    - Transcrição para RNA
    - Retrotranscrição (RNA → DNA)
    - Tradução para proteína com a tabela de tradução 

    :param seq: Sequência de DNA ou RNA a transformar.
    :type seq: Bio.Seq.Seq ou str
    :param tabela_trad: Tabela de tradução a usar (padrão "Bacterial").
    :type tabela_trad: str
    :return: None. Os resultados são apresentados no ecrã.
    :rtype: None

    :example:
    >>> from Bio.Seq import Seq
    >>> minha_seq = Seq("ATGCGTATGTTAG")
    >>> transformar_seq(minha_seq, tabela_trad="Bacterial")
    Sequência:
    ATGCGTATGTTAG
    Sequência complementar:
    TACGCATACAATC
    Sequência complementar reversa:
    CTAACATACGCAT
    Transcrição:
    AUGCGUAUGUUAG
    Transcrição reversa:
    ATGCGTATGTTAG
    Tradução:
    MRTL
    """
    
    transformacoes = {
        "Sequência": seq,
        "Sequência complementar": seq.complement(),
        "Sequência complementar reversa": seq.reverse_complement(),
        "Transcrição": seq.transcribe(),
        "Transcrição reversa": seq.back_transcribe(),
        "Tradução": seq.translate(table=tabela_trad)
    }

    for titulo, resultado in transformacoes.items():
        print(f"\n{titulo}:")
        print(resultado)


def analise_seq(seq: Seq) -> None:

    """
    Analisa uma sequência nucleotídica, fornecendo métricas básicas de exploração e qualidade biológica.

    Esta função calcula:
    - O comprimento da sequência em nucleótidos.
    - A composição de cada nucleótido (A, T, G, C).
    - A percentagem de GC.
    - As posições dos codões de início e término (moldura de leitura 0).

    :param seq: Sequência de DNA ou RNA a analisar
    :type seq: Bio.Seq.Seq ou str
    :return: None (os resultados são impressos no ecrã)
    :rtype: None

    :example:
    >>> from Bio.Seq import Seq
    >>> seq = Seq("ATGCGTATGTTAG")
    >>> analise_seq(seq)
    Comprimento da sequência: 13 nucleótidos
    Composição nucleotídica: Counter({'A': 5, 'T': 4, 'G': 3, 'C': 1})
    GC% = 30.77
    Codões de início encontrados em: 0 3 6
    Codões de término encontrados em: 9
    """

    comprimento = len(seq)

    comp = Counter(seq)
    
    gc = (comp.get("G", 0) + comp.get("C", 0)) / comprimento * 100
    
    start_codons = ['TTG', 'CTG', 'ATT', 'ATC', 'ATA', 'ATG', 'GTG']
    stop_codons = ['TAA', 'TAG', 'TGA']
    
    codoes_inicio = [i for i in range(0, len(seq)-2, 3) if seq[i:i+3] in start_codons]
    codoes_fim = [i for i in range(0, len(seq)-2, 3) if seq[i:i+3] in stop_codons]
    
    print(f"Comprimento da sequência: {comprimento} nucleótidos")
    print("Composição nucleotídica:", comp)
    print(f"GC% = {gc:.2f}")
    print("Codões de início encontrados em:", *codoes_inicio)
    print("Codões de término encontrados em:", *codoes_fim)


def analise_orfs_e_prots(seq: Seq, tamanho_min: int = 300, tabela_trad: str = "Bacterial", motivo: str = "GAATTC") -> None:
    
    """
    Identifica ORFs (Open Reading Frames) numa sequência e traduz para proteínas (com ou sem paragem no primeiro codão de stop). 
    Pesquisa também por motivos específicos na sequência original e nas ORFs.

    :param seq: Sequência de DNA a analisar.
    :type seq: Bio.Seq.Seq
    :param tamanho_min: Tamanho mínimo da ORF em nucleótidos.
    :type tamanho_min: int
    :param tabela_trad: Tabela de tradução genética utilizada.
    :type tabela_trad: str
    :param motivo: Motivo (string) a procurar na sequência e nas ORFs.
    :type motivo: str
    :return: None. Os resultados são apresentados no ecrã.
    :rtype: None

    :example:
    >>> from Bio.Seq import Seq
    >>> seq = Seq("ATGCGTATGTTAGCCGGATTCGCTAGCTAGCTAGCTAGCTAGCTAGCT")
    >>> analise_orfs_e_prots(seq, tamanho_min=30, motivo="GAATTC")
    ORFs encontrados (início, fim, tamanho_aa): [...]
    Proteína ORF1: M...
    Motivo GAATTC encontrado nas posições: [...]
    Motivo nas ORFs: {...}
    """

    orfs = []
    for frame in range(3):
        seq_fragmento = seq[frame:]
        comprimento_ajustado = len(seq_fragmento) - (len(seq_fragmento) % 3)
        seq_para_traduzir = seq_fragmento[:comprimento_ajustado]
        
        traduzida = seq_para_traduzir.translate(table=tabela_trad, to_stop=False)

        aa_seqs = str(traduzida).split("*")
        pos_acumulada = 0
        for aa in aa_seqs:
            start_idx = aa.find("M")
            
            if start_idx != -1:
                aa_valido = aa[start_idx:]
                
                if len(aa_valido) * 3 >= tamanho_min:
                    inicio = frame + (pos_acumulada + start_idx) * 3
                    fim = inicio + (len(aa_valido) * 3) + 3 
                    orfs.append((inicio, fim, len(aa_valido)))
            
            pos_acumulada += len(aa) + 1

    print(f"ORFs encontrados (início, fim, tamanho_aa): {orfs}\n")

    orfs_seq = [seq[inicio:fim] for inicio, fim, _ in orfs]

    proteinas = [orf.translate(table=tabela_trad) for orf in orfs_seq]
    proteinas_to_stop = [orf.translate(table=tabela_trad, to_stop=True) for orf in orfs_seq]

    for i in range(len(proteinas)):
        print(f"Proteína ORF{i+1} (Completa): {proteinas[i]}")
        print(f"Proteína ORF{i+1} (Até Stop): {proteinas_to_stop[i]}\n")

    posicoes = [i for i in range(len(seq)) if seq.startswith(motivo, i)]
    print(f"\nMotivo {motivo} encontrado nas posições:", posicoes)

    motivo_orfs = {f"ORF{i+1}": [j for j in range(len(orf)) if orf.startswith(motivo, j)] for i, orf in enumerate(orfs_seq)}
    print("\nMotivo nas ORFs:", motivo_orfs)


# Genbank


def consultar_genbank(protein_ids: list[str], nucleotide_id: str, email: str) -> None:
    
    """
    Consulta registros GenBank de proteínas e nucleótidos no NCBI
    e imprime informações resumidas sobre cada sequência.

    A função:
    - Define o email para a API do NCBI Entrez.
    - Obtém registros de proteínas pelos seus IDs.
    - Obtém registro de nucleótido pelo seu ID.
    - Para cada registro, imprime:
        - ID
        - Descrição (primeiros 100 caracteres)
        - Tamanho da sequência
        - Número de features
        - Organismo fonte

    :param protein_ids: Lista de IDs de proteínas a consultar.
    :type protein_ids: list[str]
    :param nucleotide_id: ID do registro de nucleótido a consultar.
    :type nucleotide_id: str
    :param email: Email a usar na API do NCBI Entrez.
    :type email: str
    :return: None. Os resultados são apresentados no ecrã.
    :rtype: None

    :example:
    >>> consultar_genbank(
    ...     protein_ids=["YP_240699.1", "YP_240698.1", "YP_240692.1"],
    ...     nucleotide_id="NC_007063",
    ...     email="utilizador@exemplo.pt"
    ... )
    Proteínas
     
    ID: YP_240699.1
    Descrição: ...
    Tamanho da sequência: 350
    Número de features: 3
    Organismo: Staphylococcus aureus

    ----------------------------------------
    Nucleótidos
     
    ID: NC_007063
    Descrição: ...
    Tamanho da sequência: 2842790
    Número de features: 2600
    Organismo: Staphylococcus aureus
    """
    
    Entrez.email = email

    print("Proteínas\n")
    with Entrez.efetch(db="protein", rettype="gb", retmode="text", id=",".join(protein_ids)) as handle:
        for seq_record in SeqIO.parse(handle, "gb"):
            print("ID:", seq_record.id)
            print("Descrição:", seq_record.description[:100], "...")
            print("Tamanho da sequência:", len(seq_record))
            print("Número de features:", len(seq_record.features))
            print("Organismo:", seq_record.annotations.get("source"))
            print()

    print("\nNucleótidos\n")
    with Entrez.efetch(db="nucleotide", rettype="gb", retmode="text", id=nucleotide_id) as handle:
        seq_record = SeqIO.read(handle, "genbank")
        print("ID:", seq_record.id)
        print("Descrição:", seq_record.description[:100], "...")
        print("Tamanho da sequência:", len(seq_record))
        print("Número de features:", len(seq_record.features))
        print("Organismo:", seq_record.annotations.get("source"))
        print()


def detalhar_genbank_protein(protein_id: str, email: str) -> None:
    
    """
    Obtém e imprime detalhes completos de um registro GenBank de proteína.

    A função:
    - Lê um registro GenBank de proteína do NCBI usando o ID fornecido.
    - Imprime informações gerais (ID, name, descrição, comprimento, organismo, fonte).
    - Lista referências bibliográficas associadas ao registro.
    - Lista todas as features, suas localizações e qualifiers.
    - Imprime a sequência completa de aminoácidos.

    :param protein_id: ID do registro de proteína a consultar.
    :type protein_id: str
    :param email: Email a usar na API do NCBI Entrez.
    :type email: str
    :return: None. Os resultados são apresentados no ecrã.
    :rtype: None

    :example:
    >>> detalhar_genbank_protein("YP_240699.1", email="utilizador@exemplo.pt")
    ID: YP_240699.1
    Name: YP_240699
    Description: endolysin [Staphylococcus phage 88]
    Sequence length: 481
    Organism: Staphylococcus phage 88
    Source: Staphylococcus phage 88
    REFERENCES:
     Authors: ...
     Title: ...
     Journal: ...
    FEATURES:
     Type: source
     Location: [0:481]
       organism: Staphylococcus phage 88
    SEQUENCE:
    MKKL...
    """

    Entrez.email = email

    with Entrez.efetch(
        db="protein",
        rettype="gb",
        retmode="text",
        id=protein_id
    ) as handle:
        seq_record = SeqIO.read(handle, "genbank")

    print("\nINFORMAÇÃO GERAL:")
    print("ID:", seq_record.id)
    print("Name:", seq_record.name)
    print("Description:", seq_record.description)
    print("Sequence length:", len(seq_record.seq))
    print("Organism:", seq_record.annotations.get("organism"))
    print("Source:", seq_record.annotations.get("source"))

    print("\nREFERENCES:")
    for ref in seq_record.annotations.get("references", []):
        print(" Authors:", ref.authors)
        print(" Title:", ref.title)
        print(" Journal:", ref.journal)
        print()

    print("\nFEATURES:")
    for feature in seq_record.features:
        print(" Type:", feature.type)
        print(" Location:", feature.location)

        if feature.qualifiers:
            for key, value in feature.qualifiers.items():
                print(f"  {key}: {value}")
        print()

    print("\nSEQUENCE:")
    print(seq_record.seq)



def detalhar_genbank_nucleotide(nucleotide_id: str, email: str) -> None:
    
    """
    Obtém e imprime detalhes completos de um registro GenBank de nucleótido.

    A função:
    - Lê um registro GenBank do NCBI usando o ID fornecido.
    - Imprime informações gerais (ID, name, descrição, comprimento, organismo, fonte, tipo de molécula, topologia, taxonomia, data).
    - Lista referências bibliográficas associadas ao registro.
    - Lista todas as features, suas localizações e qualifiers.
    - Imprime a sequência completa.

    :param nucleotide_id: ID do registro de nucleótido a consultar.
    :type nucleotide_id: str
    :param email: Email a usar na API do NCBI Entrez.
    :type email: str
    :return: None. Os resultados são apresentados no ecrã.
    :rtype: None

    :example:
    >>> detalhar_genbank_nucleotide("NC_007063", email="utilizador@exemplo.pt")
    ID: NC_007063
    Name: CP000253
    Description: Staphylococcus aureus subsp. aureus ST88, complete genome...
    Sequence length: 2842790
    Organism: Staphylococcus aureus subsp. aureus ST88
    Source: Staphylococcus aureus
    Molecule type: DNA
    Topology: circular
    Taxonomy: ['Bacteria', 'Firmicutes', 'Bacilli', 'Bacillales', 'Staphylococcaceae', 'Staphylococcus']
    Date: 2006-11-20
    REFERENCES:
     Authors: Smith J, et al.
     Title: Complete genome sequence...
     Journal: J Bacteriol 2006...
    FEATURES:
     Type: gene
     Location: [100:900]
       locus_tag: SA_ST88_0001
    SEQUENCE:
    ATGCGTATGTTAG...
    """
    Entrez.email = email

    with Entrez.efetch(db="nucleotide", rettype="gb", retmode="text", id=nucleotide_id) as handle:
        seq_record = SeqIO.read(handle, "genbank")

    print("\nINFORMAÇÃO GERAL:")
    print("ID:", seq_record.id)
    print("Name:", seq_record.name)
    print("Description:", seq_record.description)
    print("Sequence length:", len(seq_record))
    print("Organism:", seq_record.annotations.get("organism"))
    print("Source:", seq_record.annotations.get("source"))
    print("Molecule type:", seq_record.annotations.get("molecule_type"))
    print("Topology:", seq_record.annotations.get("topology"))
    print("Taxonomy:", seq_record.annotations.get("taxonomy"))
    print("Date:", seq_record.annotations.get("date"))

    print("\nREFERENCES:")
    for ref in seq_record.annotations.get("references", []):
        print(" Authors:", ref.authors)
        print(" Title:", ref.title)
        print(" Journal:", ref.journal)
        print()

    print("\nFEATURES:")
    for feature in seq_record.features:
        print(" Type:", feature.type)
        print(" Location:", feature.location)
        if feature.qualifiers:
            for key, value in feature.qualifiers.items():
                print(f"  {key}: {value}")
        print()

    print("\nSEQUENCE:")
    print(seq_record.seq)


# BioBLAST


def genbank_features(email: str, accession: str, locus_tags: list[str], print_features: bool = True) -> pd.DataFrame:
    
    """
    Obtém um registo GenBank do NCBI, filtra features por locus_tag e devolve os resultados num DataFrame.

    :param email: Email a usar na API do NCBI Entrez.
    :type email: str
    :param accession: Accession number do GenBank.
    :type accession: str
    :param locus_tags: Lista de locus_tag a filtrar.
    :type locus_tags: list[str]
    :param print_features: Se True, imprime os qualifiers das features encontradas.
    :type print_features: bool
    :return: DataFrame com as features filtradas e respetivos qualifiers.
    :rtype: pandas.DataFrame
    
    :example:
    >>> from Bio import Entrez
    >>> import pandas as pd
    >>> df = genbank_features(
    ...     email="utilizador@exemplo.pt",
    ...     accession="NC_007063",
    ...     locus_tags=["ST88ORF033", "ST88ORF006"]
    ... )
    >>> df.head()
    """

    Entrez.email = email

    with Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text") as handle:
        record = SeqIO.read(handle, "genbank")

    rows = []
    for feature in record.features:
        tag = feature.qualifiers.get("locus_tag", [None])[0]

        if tag in locus_tags:
            if print_features:
                print("\nFeature:", feature.type)
                for key, value in feature.qualifiers.items():
                    print(f"\n  /{key} = {value}")

            row = {
                "feature_type": feature.type, 
                "start": int(feature.location.start), 
                "end": int(feature.location.end)
            }

            for key, value in feature.qualifiers.items():
                row[key] = ";".join(value)

            rows.append(row)

    return pd.DataFrame(rows)


def analisar_blast_xml(file: str, eval_thresh: float = 1e-100, max_hits: int = 5, print_seq: bool = True) -> None:
    
    """
    Analisa resultados de BLAST em formato XML e imprime um resumo dos alinhamentos.

    A função:
    - Lê um ficheiro BLAST XML.
    - Itera sobre cada consulta (query) e apresenta o número de alinhamentos.
    - Para cada alinhamento, imprime os detalhes do hit (definição, accession, comprimento).
    - Para cada HSP, imprime o alinhamento se o E-value for inferior ao limiar especificado.
    - Limita a impressão aos primeiros `max_hits` alinhamentos de cada query.
    - Opcionalmente, imprime os primeiros 100 nucleótidos/aminoácidos do HSP (`query`, `match`, `sbjct`).

    :param file: Caminho para o ficheiro BLAST XML.
    :type file: str
    :param eval_thresh: Limiar de E-value para exibir alinhamentos.
    :type eval_thresh: float
    :param max_hits: Número máximo de hits a imprimir por query.
    :type max_hits: int
    :param print_seq: Se True, imprime os primeiros 100 caracteres das sequências do HSP.
    :type print_seq: bool
    :return: None. Os resultados são apresentados no ecrã.
    :rtype: None

    :example:
    >>> analisar_blast_xml("my_blast_5133735.xml", eval_thresh=0.01, max_hits=3, print_seq=False)
    Consulta: sp|P12345|Example_protein
    Número de alinhamentos: 10
    Hit: Example protein ABC
    Accession: ABC123
    Comprimento: 250
    Alinhamento
    E-value: 1e-20
    Score: 200
    Identidade: 180
    """
    
    with open(file) as result_handle:
        blast_records = NCBIXML.parse(result_handle)

        for blast_record in blast_records:
            print("Consulta:", blast_record.query)
            print("Número de alinhamentos:", len(blast_record.alignments))

            for alignment in blast_record.alignments[:max_hits]:
                print("\nHit:", alignment.hit_def)
                print("Accession:", alignment.accession)
                print("Comprimento:", alignment.length)

                hsp_count = 1
                for hsp in alignment.hsps:
                    if hsp.expect < eval_thresh:
                        print(f"Alinhamento {hsp_count}")
                        print("E-value:", hsp.expect)
                        print("Score:", hsp.score)
                        print("Identidade:", hsp.identities)

                        if print_seq:
                            print(hsp.query[:100] + "...")
                            print(hsp.match[:100] + "...")
                            print(hsp.sbjct[:100] + "...")
                        hsp_count += 1


# Filogenia


def extrair_genes(gb_file: str, ids_hits: list[str], palavra_chave: str, email: str, output_fasta: str = "sequencias_hits.fasta") -> list[SeqRecord]:
    
    """
    Extrai genes de um registro GenBank e de uma lista de hits no NCBI cujo produto ou locus_tag contenha a palavra-chave especificada.
    Salva os genes encontrados num ficheiro FASTA e retorna a lista de SeqRecords.

    :param gb_file: Caminho para o arquivo GenBank original a analisar.
    :type gb_file: str
    :param ids_hits: Lista de IDs de hits nucleotídicos no NCBI.
    :type ids_hits: list[str]
    :param palavra_chave: Palavra-chave a procurar nos produtos ou locus_tags.
    :type palavra_chave: str
    :param email: Email a usar na API do NCBI Entrez.
    :type email: str
    :param output_fasta: Nome do ficheiro de output FASTA.
    :type output_fasta: str
    :return: Lista de SeqRecords correspondentes aos genes encontrados.
    :rtype: list[Bio.SeqRecord.SeqRecord]

    :example:
    >>> sequencias = extrair_genes(gb_file="5133735.gb", ids_hits=["NC_007063", "ON571632"], palavra_chave="lysin", email="utilizador@exemplo.pt", output_fasta="hits_lysin.fasta")
    >>> len(sequencias)
    3
    """
    
    Entrez.email = email
    sequencias_encontradas = []
    palavra_chave = palavra_chave.lower()

    try:
        meu_registo = SeqIO.read(gb_file, "genbank")
        for feat in meu_registo.features:
            if feat.type == "CDS":
                product = feat.qualifiers.get("product", [""])[0].lower()
                locus = feat.qualifiers.get("locus_tag", [""])[0].lower()
                
                if palavra_chave in product or palavra_chave in locus:
                    novo_record = SeqRecord(
                        feat.extract(meu_registo.seq),
                        id=f"Local_{meu_registo.id}",
                        description=f"Produto: {product}"
                    )
                    sequencias_encontradas.append(novo_record)
                    break
    except Exception as e:
        print(f"Erro ao ler ficheiro: {e}")

    for id_hit in ids_hits:
        try:
            with Entrez.efetch(db="nucleotide", id=id_hit, rettype="gb", retmode="text") as handle:
                record_completo = SeqIO.read(handle, "genbank")
                for feature in record_completo.features:
                    if feature.type == "CDS":
                        product = feature.qualifiers.get("product", [""])[0].lower()
                        locus = feature.qualifiers.get("locus_tag", [""])[0].lower()

                        if palavra_chave in product or palavra_chave in locus:
                            hit_record = SeqRecord(
                                feature.extract(record_completo.seq),
                                id=id_hit,
                                description=product
                            )
                            sequencias_encontradas.append(hit_record)
                            break
        except Exception as e:
            print(f"Erro ao obter ID {id_hit} do NCBI: {e}")

    if sequencias_encontradas:
        SeqIO.write(sequencias_encontradas, output_fasta, "fasta")
        print(f"Sucesso: {len(sequencias_encontradas)} sequências gravadas em {output_fasta}")

    return sequencias_encontradas


def avaliar_alinhamento(file_alignment: str, format_alignment: str = "clustal") -> list[tuple[str, str, float]]:

    """
    Avalia um alinhamento múltiplo de sequências, calculando a identidade entre todos os pares de sequências.
    Retorna uma lista de tuplos com os pares de sequências e o score de identidade em percentagem, ordenada de forma decrescente.

    :param file_alignment: Caminho para o arquivo de alinhamento.
    :type file_alignment: str
    :param format_alignment: Formato do alinhamento (padrão "clustal")
    :type format_alignment: str
    :return: Lista de tuplos ordenada decrescentemente.
    :rtype: list[tuple[str, str, float]]

    :example:
    >>> scores = avaliar_alinhamento("tail.aln-clustal_num")
    >>> scores[:5]
    [('seq1', 'seq3', 98.5), ('seq2', 'seq4', 97.3), ...]
    """

    alignment = AlignIO.read(file_alignment, format_alignment)
    print(f"Número de sequências: {len(alignment)}")
    print(f"Comprimento do alinhamento: {alignment.get_alignment_length()} bp\n")

    for record in alignment:
        print(f"{record.id} - {record.seq}")

    calculator = DistanceCalculator('identity')
    dm = calculator.get_distance(alignment)
    lista_scores = []

    for i, name1 in enumerate(dm.names):
        for j, name2 in enumerate(dm.names):
            if i < j:
                score = (1 - dm[i, j]) * 100
                lista_scores.append((name1, name2, score))

    lista_scores.sort(key=lambda x: x[2], reverse=True)

    print("Avaliação do Alinhamento (Ordem Decrescente):\n")
    for n1, n2, s in lista_scores:
        print(f"{n1} vs {n2} | Identidade: {s:.2f}%")

    return lista_scores


def imprimir_arvore(file_tree: str, format_tree: str = "newick") -> Phylo.BaseTree.Tree:
    
    """
    Lê uma árvore filogenética a partir de um ficheiro e imprime a sua representação.
    
    :param file_tree: Caminho para o ficheiro da árvore.
    :type file_tree: str
    :param format_tree: Formato do ficheiro da árvore (padrão "newick")
    :type format_tree: str
    :return: Objeto da árvore filogenética lida
    :rtype: Bio.Phylo.BaseTree.Tree

    :example:
    >>> arvore = imprimir_arvore("tail.phylotree")
    >>> print(arvore.rooted)
    True
    """
    
    tree = Phylo.read(file_tree, format_tree)
    print(tree)
    return tree


def desenhar_arvore(tree: Phylo.BaseTree.Tree) -> Phylo.BaseTree.Tree:
    
    """
    Desenha a árvore filogenética em formato ASCII.

    :param tree: Objeto da árvore filogenética
    :type tree: Bio.Phylo.BaseTree.Tree
    :return: O objeto da árvore fornecida
    :rtype: Bio.Phylo.BaseTree.Tree

    :example:
    >>> from Bio import Phylo
    >>> arvore = Phylo.read("tail.phylotree", "newick")
    >>> desenhar_arvore(arvore)
    """

    Phylo.draw_ascii(tree)
    return tree


# Variantes


def analisar_variantes(nome_proteina, dna_fago88, dna_homologo):

    """
    Analisa e compara variantes proteicas a partir de duas sequências de DNA.

    Esta função limpa as sequências de entrada, traduz para proteína utilizando a tabela genética bacteriana (11) e imprime as mutações (substituições) encontradas posição a posição.

    :param nome_proteina: O nome da proteína em análise (ex: 'Holina')
    :type nome_proteina: str
    :param dna_fago88: A sequência de nucleótidos do fago em estudo
    :type dna_fago88: str
    :param dna_homologo: A sequência de nucleótidos do fago homólogo para comparação
    :type dna_homologo: str
    :return: None (A função imprime os resultados diretamente no terminal)
    :rtype: None

    :example:
    >>> dna_holi_fago88 = "ATGGATATTAACT-GGAA-ATTGAGATTCAAAAACAAAGCAGTACTAACT----GGTTTAGTTGGAGCATTGTTGCTATTTATCAAGCAAGTCAC----GGATTTATTCGGATTAGATTTATCTACTCAATTAAATCAAGCTAGCGCAATTATAGGCGCTATCCTCACGTTACTTACAGGTATTGGCGTTATTACTGACCCAAC-------GTCAAAAGGCGTCTCAGATTCATCTATAGCACAG--ACATATCAAGCGCCTAGAGATAGCAATAAAGAAGAACAACAAGTTACGT-GGAAATCATCACAAGACAGCAGTTTAACGCCGGAATTAAGCACGAAAGCACCAAAAG-------AAT---ATGATACAT---CACAACCTTTCACA-GACGCCTCTAACGATGTTGGCTTTGATGTGAA------------------TGAGTATCATCATGGAGGTGGCGACAATGCAAGCAAAATTAACTAA"
    >>> dna_holi_homologo = "ATGGATATTAACT-GGAA-ATTGAGATTCAAAAACAAAGCAGTACTAACT----GGTTTAGTTGGAGCATTGTTGCTATTTATCAAGCAAGTCAC----GGATTTATTCGGATTAGATTTATCTACTCAATTAAATCAAGCTAGCGCAATTATAGGCGCTATCCTCACGTTACTTACAGGTATTGGCGTTATTACTGACCCAAC-------GTCAAAAGGCGTCTCAGATTCATCTATAGCACAG--ACATATCAAGCGCCTAGAGATAGCAATAAAGAAGAACAACAAGTTACGT-GGAAATCATCACAAGACAGCAGTTTAACGCCGGAATTAAGCACGAAAGCACCAAAAG-------AAT---ATGATACAT---CACAACCTTTCACA-GACGCCTCTAACGATGTTGGCTTTGATGTGAA------------------TGAGTATCATCATGGAGGTGGCGACAATGCAAGCAAAATTAACTAA"
    >>> analisar_variantes("Holina", dna_holi_fago88, dna_holi_homologo)
    Análise de variantes da Holina: 

    Proteína do nosso fago: MDINWKLRFKNKAVLTGLVGALLLFIKQVTDLFGLDLSTQLNQASAIIGAILTLLTGIGVITDPTSKGVSDSSIAQTYQAPRDSNKEEQQVTWKSSQDSSLTPELSTKAPKEYDTSQPFTDASNDVGFDVNEYHHGGGDNASKIN*
    Proteína do fago homólogo: MDINWKLRFKNKAVLTGLVGALLLFIKQVTDLFGLDLSTQLNQASAIIGAILTLLTGIGVITDPTSKGVSDSSIAQTYQAPRDSNKEEQQVTWKSSQDSSLTPELSTKAPKEYDTSQPFTDASNDVGFDVNEYHHGGGDNASKIN*

    Tamanho da proteína do fago 88: 146 aminoácidos 

    Foram encontradas 0 mutações.
    """

    print(f"Análise de variantes da {nome_proteina}: \n")
    
    def limpar(seq):
        s = seq.replace("-", "").replace("\n", "").replace(" ", "")
        return s[:(len(s)//3)*3] 

    prot_fago88 = Seq(limpar(dna_fago88)).translate(table=11)
    print(f"Proteína do nosso fago: {prot_fago88}")
    prot_fago_homol = Seq(limpar(dna_homologo)).translate(table=11)
    print(f"Proteína do fago homólogo: {prot_fago_homol}\n")
    
    tamanho = len(prot_fago88)
    print(f"Tamanho da proteína do fago 88: {tamanho - 1} aminoácidos \n")

    mutacoes = 0
    for i in range(min(len(prot_fago88), len(prot_fago_homol))):
        aa1 = prot_fago88[i]
        aa2 = prot_fago_homol[i]
        
        if aa1 != aa2:
            print(f"Posição {i+1}: {aa1} passou a ser {aa2}")
            mutacoes += 1
    
    if mutacoes > 1 or mutacoes == 0:
        print(f"Foram encontradas {mutacoes} mutações.")
    else:
        print(f"Foi encontrada {mutacoes} mutação.")