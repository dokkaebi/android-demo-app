package org.pytorch.imagesegmentation;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;


public class MainActivity extends AppCompatActivity implements Runnable {
    private ImageView mImageView;
    private Button mButtonSegment;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
    private String mImagename = "rgb_rawlight.png";

    // see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of classes with indexes
    private static final int CLASSNUM = 21;
    private static final int DOG = 12;
    private static final int PERSON = 15;
    private static final int SHEEP = 17;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename));
            mBitmap = Bitmap.createScaledBitmap(mBitmap, 640, 384, true);
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);

        final Button buttonRestart = findViewById(R.id.restartButton);
        buttonRestart.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if (mImagename == "deeplab.jpg")
                    mImagename = "dog.jpg";
                else
                    mImagename = "deeplab.jpg";
                try {
                    mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename));
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("ImageSegmentation", "Error reading assets", e);
                    finish();
                }
            }
        });


        mButtonSegment = findViewById(R.id.segmentButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonSegment.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonSegment.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonSegment.setText(getString(R.string.run_model));

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        try {
//            String fp = MainActivity.assetFilePath(getApplicationContext(), "mobilenet_mobile.pt");
//            String fp = MainActivity.assetFilePath(getApplicationContext(), "hrnet_mobile.pt");
            String fp = MainActivity.assetFilePath(getApplicationContext(), "dlab_mobile.pt");
            mModule = LiteModuleLoader.load(fp);
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        } catch (Exception e) {
            Log.e("ImageSegmentation", "Error", e);
            finish();
        }
    }

    @Override
    public void run() {
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        final float[] inputs = inputTensor.getDataAsFloatArray();

        // warm up
        IValue outValue = mModule.forward(IValue.from(inputTensor));
        final long startTime = SystemClock.elapsedRealtime();
        for (int i = 0; i < 100; i++) {
            outValue = mModule.forward(IValue.from(inputTensor));
        }
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("ImageSegmentation",  "inference time (ms): " + inferenceTime);
        Log.d("ImageSegmentation", "output " + outValue.toString());

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mButtonSegment.setEnabled(true);
                mButtonSegment.setText("took " + inferenceTime + " ms");
                mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            }
        });
    }
}
