from Bio import SeqIO

fasta_path = "/opt/data/latent/data/dna/chr03.fa"

records = list(SeqIO.parse(fasta_path, "fasta"))
new_seq = records[0].seq[:100]

records[0].seq = new_seq

SeqIO.write(records, "b1.fasta", "fasta")

b1_read = list(SeqIO.parse("b1.fasta", "fasta"))

print("done")
