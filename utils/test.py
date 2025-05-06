# TEST FILE

############################################################################################################
################################################# STEP 2.a #################################################


# 1- To check if the download of the image works
# Create output dir if needed
os.makedirs('./outputs', exist_ok=True)

# Get first batch
first_batch = next(iter(dataloader_cs_train))
# I need filenames to save the images, but i don't think we need it when doing training
# The idea is to save some images and masks for the report (maybe we can understand if the image is so much worse at the first
# epochs is so much worse that at the 50 and discuss some comparison in the report)
images, masks, filenames = first_batch

# Number of samples you want to save from the batch
num_to_save = min(5, len(images))  # e.g., save 5 or fewer

for i in range(num_to_save):
    img_tensor = images[i]
    mask_tensor = masks[i]

    # Check the pixel values of the first mask in the batch, each value should be in the range [0, 18]? (bc 19 classes) + 255 for void
    mask = mask_tensor.cpu().numpy()  # Convert mask tensor to NumPy array

    # Show the unique class values in the mask
    print(f"Unique class values in the mask: {np.unique(mask)}")

    img_pil = TF.to_pil_image(img_tensor.cpu())
    # Convert mask tensor to PIL image, i am using long int64 to keep the class labels but image fromarray doen't support them
    mask_pil = Image.fromarray(mask_tensor.byte().cpu().numpy())  # Convert to uint8 before Image.fromarray

    base_filename = filenames[i].replace("leftImg8bit", "")
    img_path = f'./outputs/{base_filename}_image.png'
    mask_path = f'./outputs/{base_filename}_mask.png'

    img_pil.save(img_path)
    mask_pil.save(mask_path)

    print(f"Saved image to {img_path}")
    print(f"Saved mask to {mask_path}")

# 2- TRYING OUT COMPUTE_MIOU
print("************trying out compute_miou:***************")
# Dummy ground truth and prediction with 3 classes 
gt_images = [
np.array([
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2]
]),
np.array([
    [2, 2, 2],
    [1, 1, 1],
    [0, 0, 0]
])]
pred_images = [
np.array([
    [0, 1, 2],   # correct
    [0, 0, 2],   # 1 → 0 (mistake)
    [0, 1, 1]    # 2 → 1 (mistake)
]),
np.array([
    [2, 1, 2],   # 2 → 1 (mistake)
    [1, 1, 1],   # correct
    [0, 1, 0]    # 1 → 0 (mistake)
])]

mean_iou_dummy, iou_per_class_dummy, intersections_dummy, unions_dummy = compute_miou(gt_images, pred_images, num_classes=3)
print("__________dummy try_________")
print("Mean IoU:", mean_iou_dummy)
print("IoU per class:", iou_per_class_dummy)
print("Intersections:", intersections_dummy)
print("Unions:", unions_dummy)

print("_________try with saved masks________")
gt_mask = np.array(Image.open("outputs/monchengladbach_000000_019500_.png_mask.png").convert("L"))
#gt_mask = np.array(Image.open("outputs/hanover_000000_006922_.png_mask.png").convert("L"))
print("GT labels:", np.unique(gt_mask))
valid_mask = gt_mask != 255
num_classes = int(np.max(gt_mask[valid_mask]) + 1)
mean_iou, iou_per_class, intersections, unions = compute_miou(gt_mask, gt_mask, num_classes)
print(f"mean iou = {mean_iou}")
print(f"iou per class= {iou_per_class}")
print("Intersections:", intersections)
print("Unions:", unions)