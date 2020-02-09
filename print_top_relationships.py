"""
Given results/* files, print out the top k object relationships in each test image


"""
import argparse
from os import path
import torch


def main():
    parser = argparse.ArgumentParser(description="Print top k object relationships, based on relationship scores")
    parser.add_argument("-k", "--top_k", type=int, default=10, help="Print top k object relationships")
    parser.add_argument("-n", "--n_images", type=int, default=0, help="Number of test images to print")
    parser.add_argument("--results_dir", type=str, default='./results')
    args = parser.parse_args()

    
    # args.top_k
    # args.n_images
    predicates = torch.load(path.join(args.results_dir, "predictions_pred.pth"))
    objects = torch.load(path.join(args.results_dir, "predictions.pth"))
    descript = torch.load(path.join(args.results_dir, "test_description.pth"))

    if args.n_images > 0:
        print("Showing first {} images".format(args.n_images))
        objects = objects[:args.n_images]
        predicates = predicates[:args.n_images]
        
    ct = 0
    for iobjs, ipreds in zip(objects, predicates):
        # find the location of the top k scores
        top_rel_scores, top_pred_idxs = ipreds.get_field("scores").max(1)
        top_pred_scores, top_rel_idxs = top_rel_scores.topk(args.top_k)
        print("\n\n\n##################\nImage {}\n##################".format(ct))
        for score, rel_idx in zip(top_pred_scores, top_rel_idxs):
            rel_pair = ipreds.get_field("idx_pairs")[rel_idx]
            # here's a tricky part: we have to look up the global object label from the local label
            subject = descript.description["ind_to_classes"][iobjs.get_field("labels")[rel_pair[0].item()].item()]
            target = descript.description["ind_to_classes"][iobjs.get_field("labels")[rel_pair[1].item()].item()]
            predicate = descript.description["ind_to_predicates"][top_pred_idxs[rel_idx]]
            print("{:12}{:12}{:12} score {:.3f}".format(subject, predicate, target, score))

        ct += 1
            
    
if __name__ == "__main__":
    main()
