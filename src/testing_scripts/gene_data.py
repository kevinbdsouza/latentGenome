from Bio import SeqIO

fasta_dir = "/data2/latent/data/dna/"

chr = 21

fasta_path = fasta_dir + "chr" + str(chr) + ".fa"

records = list(SeqIO.parse(fasta_path, "fasta"))
new_seq = records[0].seq[:100]

records[0].seq = new_seq

SeqIO.write(records, "b1.fasta", "fasta")

b1_read = list(SeqIO.parse("b1.fasta", "fasta"))

print("done")
